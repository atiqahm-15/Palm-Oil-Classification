# Palm Ripeness app design: decisions

> Snapshot from the Claude project note `claude/app-design-notes.md`, as of 2026-09-21. Screen specs are in `DESIGN_HANDOFF.md` in this folder.

Design canvas: https://claude.ai/artifact/FaAA3dueVpg2UDKrsgQunK (private, owner only)

## Screens (360x800, Android)

Main flow: Home, Camera, Review (the Preview step in the report), Results.
States: Home (model not loaded), Camera (access off), Camera (low light), Results (analysing), Results (unreadable photo), Results (close call).

## Decisions

- Layout follows the report section 3.2 flow: Home, Preview, Result. Copy: Take photo, Upload image, Detect ripeness, Retake, New photo.
- Result screen shows the stage, the model confidence, and a per-stage column chart so a close call (for example 51% ripe vs 45% underripe) is visible.
- Status colours follow the FYP1 mockup convention: unripe red, underripe amber, ripe green, overripe brown. Stage name is always written out next to the colour (red/green colour-blindness).
- Buttons are dark ink, not green, so green only ever means "ripe".
- No harvest advice chips: the app reports the model output only.
- Wordmark only, no logo. FYP1 used a leaf icon and the name "Palm Oil Ripeness Detector"; not carried over.
- **Exception, needs your sign-off:** "no logo" cannot extend to the launcher icon, because Android
  has to draw *something* on the home screen. `ic_launcher_foreground.xml` is a bunch silhouette
  (the fruit, not a leaf) on `ink`. It was drawn during implementation, not specified, so it is the
  one visual asset in the app that no one asked for. Keep it, or replace it with the "PR" wordmark
  letterform, or supply your own.
- Fonts: Fraunces (headings, verdict) and Instrument Sans (UI).

## Decisions made during implementation (2026-09-24)

These came out of building the screens rather than drawing them, and they change the spec in
`DESIGN_HANDOFF.md` where they disagree with it.

- **The Analysing screen loses its percentage.** The handoff called for a determinate 0-100 bar.
  Inference is tens of milliseconds, so there is no progress to report and the bar would be
  fiction. It is now an indeterminate spinner held for a **400 ms floor**, so the step registers
  rather than flickers. Closes the second open question below.
- **Low light is measured as mean luma of the preview Y plane**, smoothed, with separate entry and
  exit thresholds so the hint does not chatter at the boundary. The numbers (enter below 50, clear
  above 70, on the 0-255 scale) are a starting point and have **not** been calibrated against field
  photographs; the third open question stays open until they are checked on a phone.
- **Capture is bound to the preview viewport.** Without this, a full-bleed preview on a tall
  screen shows a slice of the sensor field while the saved JPEG contains the rest of it too, and
  the user would frame one image while the model is fed another.
- **The primary button gains an on-dark variant.** Review puts "Detect ripeness" on `cam`, where
  the standard ink fill would disappear, so the fill and text swap to ground on ink.
- **minSdk is 26.** Google Fonts publishes both faces only as variable fonts, and setting a weight
  on the `wght` axis needs API 26. Fraunces' default instance is Thin, so on anything older every
  heading in the app would render hairline.

## Resolved 2026-09-25: the capture is cropped to the brackets

The brackets cover 280 x 360 dp of a 360 x 800 dp full-bleed preview, so a bunch that fills them
occupies roughly a third of the frame. The model was fed the **whole** frame squashed to 224 x 224,
while every image it was trained on is a tight crop around a single bunch — so following the hint
handed the model a smaller, more background-heavy bunch than anything in training. Invisible in the
parity fixtures, because those are already crops.

Re-scoring the full test split through simulated framings put a number on it: **-45.7 points** at
the app's geometry (`results/framing_shift/`). Of the three ways out — leave it, crop to the
brackets, or widen the brackets to the frame — the measurement ruled out leaving it, and **crop to
the brackets** was chosen over widening them because it does not depend on the user complying with
a hint: whatever they frame, the app cuts to the box they were aiming through. Simulation also
showed why a squarer box alone would not have been enough — at the app's ~25% area fraction, a 1:1
framing loses 47.8 points, slightly *worse* than the tall one. Scale was the problem, not shape.

The brackets are therefore no longer decoration. `CameraScreen` measures them against the preview
and passes the bounds up with the JPEG; `MainViewModel` crops the decoded frame before anything
sees it, so Review shows exactly what the model gets. The hint changed from "Fill the frame with
one bunch" to "Fill the brackets with one bunch". Gallery uploads are not cropped — the app did not
frame them and has no idea where the subject sits.

Scored at the brackets' own 7:9 aspect, the crop recovers 44.0 of the 45.7 points: **92.47%**
against a 94.16% baseline (`results/framing_shift_brackets/`). The residual sits on underripe
(recall 0.904 → 0.834), the boundary that is this project's weak point everywhere else too.

Two things this does not settle, both stated in the report's limitation register (L8): the
simulated backgrounds are synthetic rather than real plantation clutter, and a bunch that overflows
the brackets is now clipped rather than merely diluted. Both wanted field photographs, which could
not be obtained.

## Placeholders to replace

- Bunch drawings (Home stage strip, viewfinder, Review, Results photo) stand in for dataset or real photos.
- Percentages on Results are sample values from the report wireframe.

## Not tested

The flash toggle has not been exercised on hardware: emulators have no torch, so
`FlashToggle` hides itself when `hasFlashUnit()` is false and the on-state styling has only been
seen in a preview. The low-light hint has not been seen trigger on a real scene either.
