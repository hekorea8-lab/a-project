---
name: Core Intelligence System
colors:
  surface: '#f8f9ff'
  surface-dim: '#cbdbf5'
  surface-bright: '#f8f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#eff4ff'
  surface-container: '#e5eeff'
  surface-container-high: '#dce9ff'
  surface-container-highest: '#d3e4fe'
  on-surface: '#0b1c30'
  on-surface-variant: '#414750'
  inverse-surface: '#213145'
  inverse-on-surface: '#eaf1ff'
  outline: '#717781'
  outline-variant: '#c1c7d1'
  surface-tint: '#16629d'
  primary: '#00416e'
  on-primary: '#ffffff'
  primary-container: '#005994'
  on-primary-container: '#a7cfff'
  inverse-primary: '#9dcaff'
  secondary: '#34618c'
  on-secondary: '#ffffff'
  secondary-container: '#a1cdfe'
  on-secondary-container: '#285782'
  tertiary: '#653100'
  on-tertiary: '#ffffff'
  tertiary-container: '#874401'
  on-tertiary-container: '#ffbe90'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#d1e4ff'
  primary-fixed-dim: '#9dcaff'
  on-primary-fixed: '#001d35'
  on-primary-fixed-variant: '#00497b'
  secondary-fixed: '#d0e4ff'
  secondary-fixed-dim: '#9fcafb'
  on-secondary-fixed: '#001d34'
  on-secondary-fixed-variant: '#164973'
  tertiary-fixed: '#ffdcc5'
  tertiary-fixed-dim: '#ffb783'
  on-tertiary-fixed: '#301400'
  on-tertiary-fixed-variant: '#703700'
  background: '#f8f9ff'
  on-background: '#0b1c30'
  surface-variant: '#d3e4fe'
  risk-high: '#DC2626'
  risk-medium: '#F59E0B'
  risk-low: '#10B981'
  surface-subtle: '#F8FAFC'
  border-muted: '#E2E8F0'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 30px
    fontWeight: '600'
    lineHeight: 38px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  title-lg:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  title-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '600'
    lineHeight: 24px
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 18px
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.02em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.04em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit: 4px
  container-margin: 24px
  gutter: 16px
  row-height-dense: 32px
  row-height-standard: 48px
---

## Brand & Style

The design system is engineered for high-stakes enterprise environments where precision and trust are paramount. It adopts a **Corporate / Modern** aesthetic with **Minimalist** influences, prioritizing data density without sacrificing clarity.

The personality is authoritative and analytical. It mimics the functional rigor of high-end business intelligence tools like Power BI, utilizing a structured information hierarchy to transform complex AI-driven tax risk data into actionable insights. The visual language uses a "Utility First" approach: every line, color, and spacing choice serves the purpose of risk identification and financial oversight.

The system targets C-level executives and tax professionals who require a "Command Center" view of their organization's fiscal health. The emotional response is one of controlled confidence and institutional reliability.

## Colors

The palette is anchored by the corporate blue derived from the brand identity, serving as the primary driver for interactive elements and brand presence.

- **Primary & Secondary:** Used for navigational anchors, primary actions, and branding.
- **Systematic Grays:** A cool-toned gray scale (Slate) is used for text hierarchy and UI surfaces to reduce eye strain during prolonged data analysis.
- **Risk Semantic Colors:** These are strictly reserved for AI-detected risk levels. **Crimson (#DC2626)** indicates critical tax discrepancies, **Amber (#F59E0B)** signifies moderate warnings, and **Emerald (#10B981)** denotes verified compliance.
- **Backgrounds:** The interface utilizes a tiered white/off-white system to separate the sidebar, global header, and the primary data workspace.

## Typography

This design system utilizes **Inter** for all primary UI and reading tasks due to its exceptional legibility in dense interfaces. **JetBrains Mono** is introduced as a secondary functional font for numerical data, financial figures, and AI confidence scores to ensure clear character differentiation and alignment in tables.

Typography scales are tight to support high information density. For mobile views, `display-lg` and `headline-lg` should be capped at 32px (`headline-md` equivalent) to prevent excessive scrolling. Use `body-sm` for secondary metadata and `label` styles for all non-prose technical markers.

## Layout & Spacing

The layout follows a **Fixed Grid** model for the primary workspace to ensure dashboard widgets maintain consistent aspect ratios for charts.

- **Grid:** A 12-column system with 16px gutters.
- **Density:** The system defaults to a "Dense" setting, utilizing a 4px base unit. Data tables should prioritize horizontal space, using narrow margins and 32px row heights.
- **Breakpoints:**
  - **Desktop (1440px+):** Full 12-column visibility with persistent left navigation.
  - **Tablet (768px - 1439px):** Reflows to an 8-column grid; navigation collapses into a rail.
  - **Mobile (<767px):** Single column stack; data tables use horizontal scrolling or cards.

## Elevation & Depth

To maintain a professional, "Power BI" aesthetic, elevation is achieved primarily through **Tonal Layers** and **Low-contrast Outlines** rather than heavy shadows.

- **Level 0 (Base):** The primary canvas color (`#F8FAFC`).
- **Level 1 (Cards/Widgets):** White background with a 1px border (`#E2E8F0`). No shadow.
- **Level 2 (Dropdowns/Modals):** White background with a subtle ambient shadow (0px 4px 12px rgba(0, 0, 0, 0.05)) to indicate interactivity and focus.
- **Separation:** Use vertical dividers (`1px`) in headers and sidebars to group functional zones without creating unnecessary visual weight.

## Shapes

The design system uses **Soft** roundedness (4px) to balance modern approachability with institutional rigidity.

- **Components:** Buttons, input fields, and dashboard cards use a 4px radius.
- **Tags/Chips:** May use a slightly larger 6px radius to distinguish them from interactive buttons.
- **Icons:** Should follow a sharp, linear style with 1.5px stroke weight to match the precision of the typography.

## Components

### Buttons
- **Primary:** Solid `#005994` with white text. High emphasis.
- **Secondary:** Ghost style with `#005994` border and text.
- **Danger:** Solid `risk-high` for destructive actions or critical overrides.

### Data Tables
- **Header:** Light gray background (`#F1F5F9`) with `label-md` bold text.
- **Cells:** `body-md` for text, `label-md` (JetBrains Mono) for financial figures.
- **States:** Highlight rows on hover with a subtle blue tint.

### Risk Scoring Cards
- Large numerical score using `display-lg`.
- A 4px vertical "risk-indicator" bar on the left edge colored by the `risk-` semantic palette.
- Sparkline charts embedded for 30-day trend visualization.

### Input Fields
- Flat design with a 1px border.
- Focus state uses a 2px primary blue halo with 20% opacity.
- AI-assistant fields should have a subtle gradient border to indicate "Active AI" status.

### AI Insight Panels
- Distinguished by a soft background tint and a specialized icon set.
- Collapsible "Accordion" style to manage vertical space in the dashboard.
