// Unit-test the actual pad controller with deterministic DOM, clock and fetch
// doubles. This does not claim browser/layout or physical-tablet verification.
const vm = require('node:vm');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const code = fs.readFileSync(0, 'utf8');
const noop = () => {};
const context2d = new Proxy({}, { get: () => noop });
function element() {
  return {value: '', innerHTML: '', textContent: '', width: 800, height: 400,
    classList: {add: noop, remove: noop}, style: {},
    getBoundingClientRect: () => ({width: 800, height: 400, left: 0, top: 0}),
    getContext: () => context2d, appendChild: noop, addEventListener: noop,
    setPointerCapture: noop, releasePointerCapture: noop};
}
const nodes = new Map();
const timers = new Map();
const requests = [];
let timerId = 0;
class Socket {
  static OPEN = 1; static CONNECTING = 0;
  constructor() { this.readyState = 1; }
  send() {}
}
const sandbox = {
  document: {getElementById: id => {
    if (!nodes.has(id)) nodes.set(id, element());
    return nodes.get(id);
  }, createElement: element},
  window: {devicePixelRatio: 1, addEventListener: noop,
    location: {protocol: 'http:', host: 'localhost'}},
  console: {log: noop, info: noop, warn: noop, debug: noop},
  performance: {now: () => 1000}, WebSocket: Socket,
  setTimeout: (fn, ms) => { const id = ++timerId; timers.set(id, {fn, ms}); return id; },
  clearTimeout: id => timers.delete(id),
  fetch: (url, options) => new Promise(resolve => requests.push({url, options, resolve})),
};
vm.createContext(sandbox);
vm.runInContext(code, sandbox);
const event = (type, extra={}) => sandbox.handleInputEvent({
  type, source: 'huion', coordinate_space: 'normalized', ...extra,
});
const result = text => ({ok: true, json: async () => ({
  recognized_text: text, corrected_text: text, confidence: .9, correction_confidence: 1,
})});
async function main() {
  event('stroke_start', {x: .1, y: .2});
  event('stroke_point', {x: .2, y: .3});
  event('stroke_end');
  assert.equal([...timers.values()].filter(t => t.ms === 350).length, 1);
  const first = sandbox.recognize();
  assert.equal(requests.length, 1);
  const sent = JSON.parse(requests[0].options.body);
  assert.equal(sent.strokes[0].length, 2, 'pen-down point must be retained');
  assert.equal(sent.strokes[0][0].x, 80);
  event('stroke_start', {x: .3, y: .4});
  requests[0].resolve(result('old partial text'));
  await first;
  assert.equal(nodes.get('recognized').value, '', 'never apply a stale reading during pen-down');
  event('stroke_point', {x: .4, y: .5});
  event('stroke_end');
  const second = sandbox.recognize();
  await sandbox.recognize();
  assert.equal(requests.length, 2, 'at most one request may be in flight');
  sandbox.clearScreen();
  requests[1].resolve(result('cleared text'));
  await second;
  assert.equal(nodes.get('recognized').value, '', 'clear must invalidate outstanding work');
  console.log('Pad controller: snapshot, auto-trigger, serialization and stale-result tests passed');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
