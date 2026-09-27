import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const name = 'DIWA__Decision_Influential_World_Abstraction_for_VLA_WAM_Policies';
const source = path.join(root, 'paper', name);
const assets = path.join(root, 'assets');
await mkdir(assets, { recursive: true });
for (const [from, to] of [
  [path.join(root, 'paper', name + '.pdf'), 'diwa-paper.pdf'],
  [path.join(source, 'main.tex'), 'main.tex'],
  [path.join(source, 'figures/architecture.jpg'), 'architecture.jpg'],
  [path.join(source, 'figures/latency.pdf'), 'latency.pdf'],
  [path.join(source, 'figures/budget_frontier.pdf'), 'budget-frontier.pdf'],
  [path.join(source, 'evidence/results_ledger.json'), 'results-ledger.json'],
  [path.join(source, 'evidence/reported_measurements.json'), 'reported-measurements.json'],
]) await copyFile(from, path.join(assets, to));

const ledger = JSON.parse(await readFile(path.join(assets, 'results-ledger.json'), 'utf8'));
const data = {
  platforms: ledger.platform_results,
  ood: ledger.ood_conditions,
  budgets: [...ledger.fixed_budget].reverse(),
  adaptive: ledger.adaptive_budget,
  ablations: ledger.ablations,
  latency: ledger.latency_ms,
};
await writeFile(path.join(assets, 'results.js'), '// Generated from the manuscript ledger by scripts/prepare-assets.mjs.\nwindow.DIWA_DATA = ' + JSON.stringify(data, null, 2) + ';\n');
console.log('Copied manuscript assets and generated browser data from the ledger.');
