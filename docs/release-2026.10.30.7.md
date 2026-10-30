# 2026.10.30.7 — language/runtime fixes

Initial base: main `97ce23b4cb5cb70de58ff16d5766f11563272958` (2026.10.30.5).
Integrated concurrent main `daa641c5e507f7ec3d7d1db7a84007fa22a695b1` (.6),
preserving scenario visit tracking and its regression. This candidate is .7.
This is a small frontend release candidate; it does not assert full site-wide
translation of legal documents or Account/admin.

## Resulting behavior

- Shared selection resolves URL, saved preference and browser locale safely.
  Regional codes normalize; inherited property names are rejected; blocked storage
  does not stop startup. Internal page and Account links keep language and anchors.
- Buy has 19 actual translated provider notices/labels and a 19-option selector.
  All 19 routes reach the already-translated SAVE instead of forcing English.
- Graph explanation, canvas labels, loading/error/source/date texts, unavailable
  ratios and financial-data warnings use 19-language dictionaries. Changing the
  language preserves the loaded/error state and does not request quotes again.
  A failed history refresh no longer leaves the old price strip as current data.
- Calculator copying, gold labels, email-preview captions, navigation and legacy
  Premium modal labels no longer fall back to English for supported languages.
- Unconfigured homepage checkout is labelled Coming soon, hides its email form
  and keeps submission disabled. No incomplete payment integration is enabled.
- Scenario keeps its validation/math and now updates the URL, navigation labels,
  description and links when changing language, including AR/HE RTL.
- Research handles blocked storage and invalid language names, localizes its UI
  navigation/metadata and keeps the receiving address and external references intact.
- Static payment-success URL now says confirmation is pending in 19 languages;
  merely opening it never claims payment or access activation.
- About describes published daily closes and delayed-data limitations accurately.
- Sunday digest form and preference service from .5 remain intact. No backend,
  email delivery contract, Terms version, analytics endpoint or payout is changed.

## Validation actually executed

- `node tests/language-release.test.cjs`: 12 checks passed, exercising real adapters
  in all 19 locales, errors, storage denial, invalid language names, provider links,
  graph loading/error preservation, Arabic numerals and URL propagation.
- `node --test tests/*.test.cjs`: 16 test files passed, including digest preference,
  subscription receipt, SAVE, calculator, freshness, scenario and research checks.
- Root `npm test`: 18 files passed; 2 auth tests blocked at import because this cloud
  checkout lacks `@simplewebauthn/server`. Not a confirmed product regression.
- Audience service `npm test`: all 4 test files passed. No real marketing messages,
  purchases, registrations or transfers were sent by these tests.
- `git diff --check`: passed. No Python changes; Python suites were not rerun.
- Real browser/physical Samsung testing was not performed in this restricted cloud
  environment. Native-language editorial review remains separate from key coverage.

## Integration and outstanding work

Push on a feature branch, with only the changed paths layered on the recorded
GitHub tree. Preserve concurrent Desktop commits. The local Desktop session owns
main/Pages integration and browser verification under the existing coordination.
The Worker APP_VERSION remains .6 because that separately deployed backend is not
modified by this frontend release. Root VERSION and frontend cache keys are .7.

Still pending: full translations of About, Terms, Privacy, Support, three articles,
Account/admin/email verification, bot VIP UI, and their server-supplied error text.
English originals remain honest English; they are not relabelled as translated.
This release must not be described as "19 languages everywhere".
