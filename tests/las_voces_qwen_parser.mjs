import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const bundle = process.env.QWEN_BUNDLE_DIR;
assert.ok(bundle, 'QWEN_BUNDLE_DIR must identify the installed Qwen Code 0.24.7 package');
const packageInfo = JSON.parse(readFileSync(resolve(bundle, 'package.json'), 'utf8'));
assert.equal(packageInfo.version, '0.24.7');
const { SubagentManager, resolveSubagentApprovalMode } = await import(
  pathToFileURL(resolve(bundle, 'chunks/chunk-AN36BHDM.js')).href
);
const project = resolve('projects/las-voces');
const canonical = JSON.parse(readFileSync(resolve(project, 'project.json'), 'utf8'));
const builder = canonical.agents.find((agent) => agent.name === 'Qwen');
const path = resolve(project, '.qwen/agents/primary-builder.md');
const parsed = new SubagentManager({}).parseSubagentContent(
  readFileSync(path, 'utf8'), path, 'project'
);
assert.deepEqual(parsed.tools, builder.tools);
assert.deepEqual(parsed.disallowedTools, builder.disallowedTools);
assert.equal(parsed.approvalMode, builder.approvalMode);
for (const [parent, expected] of [
  ['default', 'default'],
  ['auto-edit', 'auto_edit'],
  ['auto', 'auto'],
  ['yolo', 'yolo'],
  ['plan', 'default'],
]) {
  assert.equal(resolveSubagentApprovalMode(parent, parsed.approvalMode, true), expected, parent);
}
console.log('PASS: Qwen Code 0.24.7 parsed primary-builder contract');
