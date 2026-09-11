---
version: alpha
name: CyberTGN Operations Center
description: Dense dark telemetry console for local network analysts.
colors:
  primary: '#38bdf8'
  background: '#0a0e17'
  surface: '#111827'
  text: '#f3f4f6'
  muted: '#9ca3af'
  border: '#374151'
typography:
  sans:
    fontFamily: '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, sans-serif'
rounded:
  DEFAULT: '8px'
  control: '6px'
spacing:
  panel: '1.25rem'
  gap: '1.5rem'
components:
  button: {}
  panel: {}
  dialog: {}
---

## Overview

Product console for analysts monitoring a local CyberTGN instance. Existing README and API behavior supply the product context. English UI; no market-specific behavior is defined. Preserve the existing dark telemetry-console identity and compact risk meters. Favor readable operational state over decorative motion.

## Colors

Runtime CSS in `static/dashboard.html` is canonical; this document mirrors its values. Primary maps to `--accent-blue`, background to `--bg-dark`, surface to `--panel-bg`, text to `--text-main`, muted to `--text-muted`, border to `--border-color`. Semantic red, green, and amber always accompany text labels. No generated theme adapter exists.

## Typography

System sans stack for controls and data. Heading is 1.25rem, table text .85rem, metric values 1.8rem. IP addresses use code semantics.

## Layout

Twelve-column desktop grid with eight/four content split. Below 760px panels span the full width, metrics use two columns, and header actions wrap. Wide tables scroll inside their panel. Global scrollbars retain stable space and use the border/background tokens.

## Elevation & Depth

Tonal panels and thin borders define hierarchy. Only the confirmation dialog overlays the page.

## Shapes

Panels and dialog use 8px corners; action controls use 6px. Status pills retain their existing shape.

## Components

`static/dashboard.html` owns all primitives in this single-screen application. `checkedFetch` owns HTTP error/timeout handling; `showStatus` owns live operation feedback. Upload uses a native file picker plus a keyboard-accessible button and drag alternative. The app-owned HTML dialog provides modal focus, Escape and cancellation for memory reset. Busy buttons are disabled without changing labels. The same feedback rules apply to capture, reset, unblock, and upload.

Flow display retains at most 100 recent rows; counters include all received rows. Upload replaces the current display. Rendered string data is escaped. Errors preserve a retry path. Motion is disabled under reduced-motion preference.

## Do's and Don'ts

- Preserve the existing console palette and compact layout.
- Show failed actions explicitly and keep operation state honest.
- Do not claim simulated firewall actions are live enforcement.
- Do not use native browser alert/confirm dialogs or color-only status.

### Canonical UI Map

| Capability | Canonical owner | Source of truth | Allowed variants | Verification |
|---|---|---|---|---|
| Form | Upload control in static/dashboard.html | API upload contract | Native file picker and drop | Browser upload success/failure |
| Scrollbar | Global stylesheet in static/dashboard.html | CSS tokens | Panel horizontal overflow | Narrow viewport check |
| Toast | showStatus and operationStatus | docs/design.md | Persistent operation feedback | Browser failure checks |
| CRUD | checkedFetch and resetDialog | api.py routes | Reset confirmation, unblock | Browser controls and API tests |
