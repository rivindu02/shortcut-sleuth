// Dependency-free DOM state checks. Real browser rendering is checked separately.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {spawnSync} = require('node:child_process');

const python = process.env.PYTHON || 'python';
const generated = spawnSync(python, ['-c', `
import json
from shortcut_sleuth.server import Demo
d = Demo()
print(json.dumps({'catalog': d.catalog(), 'examples': {
 f'{shape}/{seed}': d.example(shape, seed) for shape in range(3) for seed in (73,74)
}}))
`], {encoding: 'utf8'});
assert.equal(generated.status, 0, generated.stderr);
const fixtures = JSON.parse(generated.stdout);
const source = fs.readFileSync('shortcut_sleuth/web/app.js', 'utf8');

class Element {
  constructor(tag='div') {
    this.tagName = tag; this.children = []; this.listeners = {}; this.attributes = {};
    this.dataset = {}; this.value = ''; this.textContent = ''; this.hidden = false;
  }
  append(...children) {this.children.push(...children);}
  replaceChildren(...children) {this.children = children;}
  setAttribute(key, value) {this.attributes[key] = value;}
  addEventListener(type, handler) {this.listeners[type] = handler;}
  click() {return this.listeners.click?.();}
  checkValidity() {return true;}
  reportValidity() {}
}

async function settle() {
  await new Promise(resolve => setImmediate(resolve));
  await new Promise(resolve => setImmediate(resolve));
}

async function app({models=false, fail=false}={}) {
  const ids = Object.fromEntries(['mode-banner','condition','error','specimen-image','object-id',
    'paired-images','model-cards','baseline-prediction','object-seed','new-object'].map(id => [id,new Element()]));
  ids.condition.value = 'correlated';
  ids['object-seed'].value = '73';
  const shapes = [0,1,2].map(i => {const e=new Element('button');e.dataset.shape=String(i);return e;});
  const backgrounds = [0,1,2].map(i => {const e=new Element('button');e.dataset.background=String(i);return e;});
  const calls = [];
  const catalog = structuredClone(fixtures.catalog);
  if (models) catalog.models = [{id:'test-cnn',architecture:'cnn',condition:'correlated',seed:0,parameters:100,protocol:'test-fixture'}];
  const context = vm.createContext({console,document:{
    getElementById: id => ids[id],
    createElement: tag => new Element(tag),
    querySelectorAll: selector => selector === '[data-shape]' ? shapes : backgrounds
  }, fetch: async url => {
    calls.push(url);
    if (fail) throw new Error('Test connection failure');
    if (url === '/api/catalog') return {ok:true,json:async()=>catalog};
    const query = new URL(url, 'http://fixture').searchParams;
    const body = structuredClone(fixtures.examples[`${query.get('shape')}/${query.get('seed')}`]);
    if (models) body.predictions['test-cnn'] = [[.8,.1,.1],[.3,.6,.1],[.6,.2,.2]];
    return {ok:true,json:async()=>body};
  }});
  vm.runInContext(source, context);
  await settle();
  return {ids,shapes,backgrounds,calls,context};
}

function text(node) {return node.textContent + node.children.map(text).join(' ');}

test('checkpoint-free mode labels missing models and renders exact server images', async () => {
  const {ids} = await app();
  assert.match(ids['mode-banner'].textContent, /No trained checkpoints/);
  assert.equal(ids.condition.disabled, true);
  assert.equal(ids['specimen-image'].src, fixtures.examples['0/73'].images[0]);
  assert.equal(ids['paired-images'].children.length, 3);
  assert.match(text(ids['model-cards']), /Awaiting a checkpoint/);
});

test('background swaps reuse the paired object and do not call inference again', async () => {
  const {ids,backgrounds,calls} = await app();
  const object = ids['object-id'].textContent;
  const requests = calls.length;
  backgrounds[1].click();
  assert.equal(ids['specimen-image'].src, fixtures.examples['0/73'].images[1]);
  assert.equal(ids['object-id'].textContent, object);
  assert.equal(calls.length, requests);
  assert.match(ids['baseline-prediction'].textContent, /Square/);
  assert.equal(backgrounds[1].attributes['aria-pressed'], 'true');
});

test('shape and seed controls request deterministic new specimens', async () => {
  const {ids,shapes} = await app();
  shapes[2].click();
  assert.equal(ids['specimen-image'].hidden, true);
  await settle();
  assert.equal(ids['specimen-image'].hidden, false);
  assert.equal(ids['specimen-image'].src, fixtures.examples['2/73'].images[0]);
  ids['new-object'].click();
  await settle();
  assert.equal(ids['object-seed'].value, 74);
  assert.equal(ids['specimen-image'].src, fixtures.examples['2/74'].images[0]);
  assert.equal(ids['new-object'].disabled, false);
});

test('loaded checkpoint controls display supplied probabilities and paired sensitivity', async () => {
  const {ids,backgrounds} = await app({models:true});
  assert.match(ids['mode-banner'].textContent, /1 real checkpoints/);
  assert.match(text(ids['model-cards']), /80.0%/);
  assert.match(text(ids['model-cards']), /Prediction changes/);
  backgrounds[1].click();
  assert.match(text(ids['model-cards']), /60.0%/);
  assert.match(text(ids['model-cards']), /50.0 pp/);
});

test('connection errors have a visible explanation', async () => {
  const {ids} = await app({fail:true});
  assert.equal(ids.error.hidden, false);
  assert.match(ids.error.textContent, /connection failure/);
  assert.match(ids['mode-banner'].textContent, /Cannot connect/);
});
