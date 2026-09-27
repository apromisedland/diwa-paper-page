import assert from 'node:assert/strict';
import { readFile, access } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

const root = fileURLToPath(new URL('../', import.meta.url));
const read = (file) => readFile(path.join(root, file), 'utf8');
const html = await read('index.html');
const links = [...html.matchAll(/(?:href|src)="([^"]+)"/g)].map((match) => match[1]);
for (const link of links) {
  assert(!link.startsWith('/') && !link.includes('C:'), `Non-relative site link: ${link}`);
  if (link.startsWith('#')) { if (link !== '#') assert(html.includes(`id="${link.slice(1)}"`), `Missing anchor: ${link}`); }
  else if (!/^https?:/.test(link)) await access(path.join(root, link.split('#')[0]));
}
const ledger = JSON.parse(await read('assets/results-ledger.json'));
const reported = JSON.parse(await read('assets/reported-measurements.json'));
const context = { window: {} };
vm.runInNewContext(await read('assets/results.js'), context);
const data = JSON.parse(JSON.stringify(context.window.DIWA_DATA));
assert.deepEqual(data.budgets, [...ledger.fixed_budget].reverse());
assert.deepEqual(data.adaptive, ledger.adaptive_budget);
assert.deepEqual(data.platforms, ledger.platform_results);
assert.deepEqual(data.ood, ledger.ood_conditions);
assert.deepEqual(data.ablations, ledger.ablations);
assert.deepEqual(data.latency, ledger.latency_ms);
const reportedBudgets = Object.entries(reported.tables).find(([key]) => key.startsWith('## 5.'))[1];
for (const point of data.budgets) {
  const row = reportedBudgets.find((entry) => entry[1] === String(point.selected_queries));
  assert(row);
  assert.equal(parseFloat(row[2]), point.standard_success_pct);
  assert.equal(parseFloat(row[3]), point.latency_ms);
}
const reportedPlatforms = Object.entries(reported.tables).find(([key]) => key.startsWith('### 2.1'))[1];
for (const point of data.platforms) {
  const row = reportedPlatforms.find((entry) => entry[0] === point.method);
  ['LIBERO', 'RoboTwin', 'RoboCasa', 'real_robot_pct', 'macro_pct'].forEach((key, index) => assert.equal(Number(point[key].toFixed(1)), parseFloat(row[index + 1])));
}
const source = 'paper/DIWA__Decision_Influential_World_Abstraction_for_VLA_WAM_Policies';
assert.equal(await read('assets/main.tex'), await read(`${source}/main.tex`));
assert.equal(await read('assets/results-ledger.json'), await read(`${source}/evidence/results_ledger.json`));
assert.equal(await read('assets/reported-measurements.json'), await read(`${source}/evidence/reported_measurements.json`));
assert.equal(data.platforms.find((row) => row.method === 'DIWA').macro_pct, 73.5);
assert.equal(data.adaptive.mean_selected_pct, 23.7);
console.log(`PASS: ${links.length} links/anchors, browser data, original source copies, all fixed budgets and platform means against both evidence records.`);
