const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.join(__dirname, '..');
const promo = require('../assets/js/loading-promo.js');

test('loading promo respects session and verified VIP preference state', () => {
  assert.equal(promo.shouldShow({sessionSeen:false, vipHidden:false, isVip:false}), true);
  assert.equal(promo.shouldShow({sessionSeen:true, vipHidden:false, isVip:false}), false);
  assert.equal(promo.shouldShow({sessionSeen:false, vipHidden:true, isVip:false}), true);
  assert.equal(promo.shouldShow({sessionSeen:false, vipHidden:true, isVip:true}), false);
  assert.equal(promo.isVipTier('VIP'), true);
  assert.equal(promo.isVipTier('premium'), true);
  assert.equal(promo.isVipTier('free'), false);
  const blocked = {getItem(){throw new Error('blocked')}, setItem(){throw new Error('blocked')}};
  assert.equal(promo.readStorage(blocked, 'x'), null);
  assert.doesNotThrow(() => promo.writeStorage(blocked, 'x', '1'));
});

test('loading promo supports every homepage language and uses the approved video', () => {
  const expected = ['en','ru','he','de','fr','es','zh','it','pt','ar','hi','ja','ko','tr','pl','nl','uk','id','vi'];
  assert.deepEqual(Object.keys(promo.copy), expected);
  const source = fs.readFileSync(path.join(root, 'assets/js/loading-promo.js'), 'utf8');
  const homepage = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
  assert.match(source, /btc2g-demo-42s-f\.mp4/);
  assert.match(source, /video\.muted = false/);
  assert.match(homepage, /assets\/js\/loading-promo\.js/);
});

test('loading promo does not block the dashboard or steal initial focus', () => {
  const source = fs.readFileSync(path.join(root, 'assets/js/loading-promo.js'), 'utf8');
  const styles = fs.readFileSync(path.join(root, 'assets/css/index.css'), 'utf8');
  assert.doesNotMatch(source, /aria-modal/);
  assert.doesNotMatch(source, /loading-promo-open/);
  assert.doesNotMatch(source, /closeButton\.focus/);
  assert.match(source, /setAttribute\('role', 'region'\)/);
  assert.match(styles, /\.loading-promo \{ position:fixed; right:18px; bottom:18px;/);
  assert.match(styles, /width:min\(220px,calc\(100vw - 16px\)\)/);
  assert.doesNotMatch(styles, /body\.loading-promo-open/);
});
