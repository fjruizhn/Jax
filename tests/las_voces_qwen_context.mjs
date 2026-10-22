// Qwen Code 0.24.7's own settings loader and model resolver are the oracle.
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

const bundle = process.env.QWEN_BUNDLE_DIR;
assert.ok(bundle, 'QWEN_BUNDLE_DIR is required');
const project = resolve('projects/las-voces');
const { loadSettings } = await import(pathToFileURL(resolve(bundle, 'chunks/chunk-ADXVOLEU.js')));
const { resolveCliGenerationConfig } = await import(pathToFileURL(resolve(bundle, 'chunks/chunk-VISH2Q33.js')));
const settings = loadSettings(project).merged;
const argv = process.env.LV001B_QWEN_MODEL ? { model: process.env.LV001B_QWEN_MODEL } : {};
const resolved = resolveCliGenerationConfig({ argv, settings,
  selectedAuthType: process.env.LV001B_QWEN_AUTH_TYPE || 'openai' });
assert.equal(resolved.generationConfig.contextWindowSize, 131072,
  'Qwen effective context differs from the project contract');
assert.equal(settings.mcpServers?.['faro-readonly']?.trust, false);
assert.deepEqual(settings.mcpServers?.['faro-readonly']?.args,
  ['authority/faro_readonly_mcp.py']);
assert.deepEqual(settings.mcpServers?.['faro-readonly']?.includeTools,
  ['skills.buscar', 'skills.leer', 'agentes.listar']);
console.log(JSON.stringify({ task: 'LV-001', qwen: '0.24.7',
  effectiveContextWindowSize: resolved.generationConfig.contextWindowSize,
  model: resolved.model, mcpServer: 'faro-readonly', scope: settings.mcpServers['faro-readonly'].scope }));
