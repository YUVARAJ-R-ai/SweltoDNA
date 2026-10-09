import { expect, test, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

const errors: string[] = [];
async function open(page: Page, hash = "") {
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  await page.goto(`/${hash}`);
  await expect(page.getByTestId("scene").locator("canvas")).toBeVisible();
}
const toSequence = async (page: Page) => { await page.keyboard.press("3"); await expect(page.getByRole("button", { name: "MAPT exon 10" })).toHaveAttribute("aria-current", "page", { timeout: 15_000 }); };

test.afterEach(() => { expect(errors, errors.join("\n")).toEqual([]); errors.length = 0; });

test("loads on the brain level with breadcrumbs and hint", async ({ page }) => {
  await open(page);
  await expect(page.getByRole("button", { name: "Brain" })).toHaveAttribute("aria-current", "page");
  await expect(page.getByText("Scroll or pinch to zoom")).toBeVisible();
});

test("#7 keyboard: zoom to the exon, move, and rewrite a base", async ({ page }) => {
  await open(page);
  await toSequence(page);
  await expect(page.getByTestId("variant-title")).toHaveText("G");
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("a");
  await expect(page.getByTestId("variant-title")).toHaveText("T → A");
  await expect(page.getByTestId("sentence")).toContainText("High impact.");
  await expect(page.getByTestId("sentence")).toContainText("activates a cryptic donor 16 bp downstream");
  await expect(page.getByTestId("consequence")).toHaveText("Exon 10 extended by 17 nt · frameshift");
  await page.keyboard.press("Control+z");
  await expect(page.getByTestId("variant-title")).toHaveText("T");
});

test("#7 virtualization keeps the DOM small anywhere in 10 kb", async ({ page }) => {
  await open(page, "#z=2");
  const ribbon = page.getByTestId("ribbon");
  await expect(page.getByTestId("chip").first()).toBeVisible();
  expect(await page.getByTestId("chip").count()).toBeLessThan(200);
  await ribbon.evaluate((el) => { el.scrollLeft = 9000 * 20; });
  await expect(page.locator('[data-testid="chip"][data-i="9000"]')).toBeAttached();
  expect(await page.getByTestId("chip").count()).toBeLessThan(200);
});

test("#7 radial switcher opens from a chip and applies the edit", async ({ page }) => {
  await open(page, "#z=2");
  await page.locator('[data-testid="chip"][data-i="5001"]').click();
  const menu = page.getByRole("menu");
  await expect(menu).toBeVisible();
  await expect(menu.getByRole("menuitem", { name: "Change to T" })).toBeDisabled();
  await menu.getByRole("menuitem", { name: "Change to A" }).click();
  await expect(page.getByTestId("variant-title")).toHaveText("T → A");
});

test("#8 dual-track tooltip reports coordinate, class, Δ and tier", async ({ page }) => {
  await open(page, "#z=2&mut");
  await expect(page.getByTestId("variant-title")).toHaveText("T → A");
  await expect(page.locator('[data-testid="chip"][data-i="5000"]')).toBeInViewport();
  const track = page.getByTestId("dual-track");
  const box = (await track.boundingBox())!;
  const first = await page.getByTestId("chip").first().getAttribute("data-i");
  const x = box.x + (5000 - Number(first)) * 20 + 10;
  await page.mouse.move(x, box.y + 20);
  await expect(page.getByTestId("track-tooltip")).toContainText("chr17:45,965,000 · Donor Loss · Δ −1.00 · High");
});

test("#9 HUD shows figures per edit and says they are simulated", async ({ page }) => {
  await open(page, "#z=2&mut");
  await page.getByRole("button", { name: "Performance" }).click();
  await expect(page.getByTestId("perf-source")).toHaveText("Simulated · updates on each edit");
});

test("saturation scan ranks the canonical donor first", async ({ page }) => {
  await open(page, "#z=2");
  await page.getByRole("button", { name: /Scan 192 variants/ }).click();
  await expect(page.getByTestId("scan-results")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("scan-results").getByRole("button").first()).toContainText("exon 10 donor +");
});

test("snapshot opens a dithered print and closes with Escape", async ({ page }) => {
  await open(page, "#z=2&mut");
  await expect(page.getByTestId("variant-title")).toHaveText("T → A");
  await page.keyboard.press("s");
  const sheet = page.getByRole("dialog", { name: "Snapshot" });
  await expect(sheet).toHaveAttribute("data-off", "false");
  await expect(sheet.getByRole("img", { name: "Dithered image" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(sheet).toHaveAttribute("data-off", "true");
});

test("no serious accessibility violations in the sequence view", async ({ page }) => {
  await open(page, "#z=2");
  await expect(page.getByTestId("variant-title")).toBeVisible();
  const r = await new AxeBuilder({ page }).exclude("[data-testid=scene]").analyze();
  const serious = r.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
  expect(serious.map((v) => `${v.id}: ${v.nodes.map((n) => `${n.target.join(" ")} → ${n.failureSummary?.split("\n").slice(-1)[0]}`).join(" | ")}`)).toEqual([]);
});
