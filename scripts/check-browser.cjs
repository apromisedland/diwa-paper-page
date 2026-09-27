const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');

(async () => {
  const root = path.resolve(__dirname, '..');
  const output = path.join(root, '.qa');
  await fs.mkdir(output, { recursive: true });
  const browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_CHANNEL ? { channel: process.env.BROWSER_CHANNEL } : {}) });
  const url = process.env.SITE_URL || 'http://127.0.0.1:4173';
  const errors = [];
  const results = [];
  try {
    for (const [name, width, height] of [['wide', 1440, 1000], ['desktop', 1024, 800], ['mobile', 390, 844], ['small', 320, 740]]) {
      const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
      page.on('pageerror', (error) => errors.push(error.message));
      page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
      await page.goto(url);
      await page.waitForSelector('.query-cell');
      await page.screenshot({ path: path.join(output, `${name}-hero.png`) });
      assert.equal(await page.locator('.query-cell').count(), 48);
      const canvasBefore = await page.locator('#flow-canvas').evaluate((canvas) => canvas.toDataURL());
      await page.waitForTimeout(150);
      const canvasAfter = await page.locator('#flow-canvas').evaluate((canvas) => canvas.toDataURL());
      assert.notEqual(canvasBefore, canvasAfter, `${name}: canvas should animate`);
      const pixels = await page.locator('#flow-canvas').evaluate((canvas) => {
        const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
        let painted = 0; for (let index = 3; index < pixels.length; index += 4) if (pixels[index]) painted++;
        return painted;
      });
      assert(pixels > 500, `${name}: blank canvas`);
      await page.locator('#motion-toggle').click();
      const paused = await page.locator('#flow-canvas').evaluate((canvas) => canvas.toDataURL());
      await page.waitForTimeout(100);
      assert.equal(paused, await page.locator('#flow-canvas').evaluate((canvas) => canvas.toDataURL()));
      const budgets = [[3, '62.9%', '61 ms'], [6, '69.8%', '71 ms'], [12, '73.6%', '91 ms'], [24, '74.1%', '132 ms'], [48, '73.4%', '213 ms']];
      for (let index = 0; index < budgets.length; index++) {
        await page.locator('#budget-slider').fill(String(index));
        const [count, success, latency] = budgets[index];
        assert.equal(await page.locator('.query-cell.selected').count(), count);
        assert.equal(await page.locator('#budget-success').textContent(), success);
        assert.equal(await page.locator('#budget-latency').textContent(), latency);
      }
      await page.locator('#adaptive-toggle').check();
      assert(await page.locator('#budget-slider').isDisabled());
      assert.equal(await page.locator('#budget-success').textContent(), '73.5%');
      assert.equal(await page.locator('#budget-latency').textContent(), '89 ms');
      assert.equal(await page.locator('.query-cell.selected').count(), 12);
      assert((await page.locator('#budget-note').textContent()).includes('23.7%'));
      await page.locator('#explore').scrollIntoViewIfNeeded();
      await page.waitForTimeout(800);
      await page.screenshot({ path: path.join(output, `${name}-budget.png`) });
      await page.locator('#adaptive-toggle').uncheck();
      await page.locator('#budget-slider').focus();
      await page.keyboard.press('Home');
      await page.keyboard.press('ArrowRight');
      assert.equal(await page.locator('#budget-success').textContent(), '69.8%');
      await page.locator('[data-result="ood"]').click();
      assert((await page.locator('#result-chart').getAttribute('aria-label')).includes('75.8 versus 65.2'));
      await page.locator('#results').scrollIntoViewIfNeeded();
      await page.waitForTimeout(650);
      await page.screenshot({ path: path.join(output, `${name}-results.png`) });
      await page.locator('[data-result="platforms"]').click();
      await page.locator('#architecture-open').click();
      assert(await page.locator('#figure-dialog').isVisible());
      await page.keyboard.press('Escape');
      assert(!(await page.locator('#figure-dialog').isVisible()));
      assert.equal(await page.evaluate(() => document.activeElement.id), 'architecture-open');
      if (width <= 520) {
        await page.locator('.menu-toggle').click();
        assert.equal(await page.locator('.menu-toggle').getAttribute('aria-expanded'), 'true');
        await page.locator('#navigation a[href="#resources"]').click();
        assert.equal(await page.locator('.menu-toggle').getAttribute('aria-expanded'), 'false');
      }
      for (const section of await page.locator('main section').all()) await section.scrollIntoViewIfNeeded();
      await page.waitForTimeout(800);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `${name}: horizontal overflow`);
      assert.equal(await page.locator('img').evaluateAll((images) => images.filter((img) => !img.complete || !img.naturalWidth || !img.alt).length), 0);
      await page.screenshot({ path: path.join(output, `${name}-full.png`), fullPage: true });
      results.push(`${name}: layout, animation pixels/pause, 5 budgets, adaptive, keyboard, filters, dialog, images passed`);
      await page.close();
    }
    const page = await browser.newPage({ reducedMotion: 'reduce' });
    await page.goto(url);
    assert(await page.locator('#motion-toggle').isDisabled());
    const still = await page.locator('#flow-canvas').evaluate((canvas) => canvas.toDataURL());
    await page.waitForTimeout(100);
    assert.equal(still, await page.locator('#flow-canvas').evaluate((canvas) => canvas.toDataURL()));
    await page.keyboard.press('Tab');
    assert.equal(await page.evaluate(() => document.activeElement.className), 'skip-link');
    const files = await page.locator('a[href]').evaluateAll((links) => [...new Set(links.map((link) => link.getAttribute('href')).filter((href) => !href.startsWith('#') && !href.startsWith('http')))]);
    for (const file of files) assert.equal((await page.request.get(new URL(file, url).href)).status(), 200, file);
    await page.close();
    const nojs = await browser.newPage({ javaScriptEnabled: false, viewport: { width: 390, height: 844 } });
    await nojs.goto(url);
    assert(await nojs.locator('#hero-title').isVisible());
    assert(await nojs.locator('.fallback-comparison').isVisible());
    assert(await nojs.locator('.resource').first().isVisible());
    assert(!(await nojs.locator('#budget-slider').isVisible()));
    await nojs.screenshot({ path: path.join(output, 'no-js.png'), fullPage: true });
    await nojs.close();
    assert.deepEqual(errors, []);
    results.push('Reduced motion, skip link, resource HTTP responses, no-JavaScript reading, and zero browser errors passed');
    await fs.writeFile(path.join(output, 'verification.json'), JSON.stringify(results, null, 2));
    console.log(results.join('\n'));
  } finally { await browser.close(); }
})().catch((error) => { console.error(error); process.exitCode = 1; });
