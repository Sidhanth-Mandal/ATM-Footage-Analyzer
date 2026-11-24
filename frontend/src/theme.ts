// Bridges the CSS design tokens (src/index.css) to code that cannot use Tailwind classes.

/** Read a CSS custom property from :root, e.g. cssColor("--color-accent"). */
export function cssColor(name: string): string {
  if (typeof document === "undefined") return "#FFFFFF";
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || "#FFFFFF";
}

/** SVG colors for the ROI editor. Use them through `style`, because presentation attributes do not resolve var(). */
export const ROI_COLOR: Record<string, string> = {
  keypad: "var(--color-roi-keypad)",
  cash_slot: "var(--color-roi-cash_slot)",
  card_slot: "var(--color-roi-card_slot)",
  atm_zone: "var(--color-roi-atm_zone)",
  ocr: "var(--color-roi-ocr)",
};

export const OVERLAY_BOX = "var(--color-overlay-box)";
