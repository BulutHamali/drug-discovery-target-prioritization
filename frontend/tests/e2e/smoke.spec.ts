import { expect, test } from "@playwright/test";

test.describe("Landing page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  test("shows Drug Target Prioritization brand", async ({ page }) => {
    await expect(page.getByRole("heading", { name: /Drug Target Prioritization/i })).toBeVisible();
  });

  test("shows tagline", async ({ page }) => {
    await expect(page.getByText(/Population genetics meets machine learning/i)).toBeVisible();
  });

  test("shows 5.59× enrichment", async ({ page }) => {
    await expect(page.locator(".enrichment-number").first()).toBeVisible();
  });

  test("Explore the targets button enters workspace", async ({ page }) => {
    await page.getByRole("button", { name: /Explore the targets/i }).click();
    await expect(page.locator(".sidebar").first()).toBeVisible();
    await expect(page.locator(".brand small").first()).toBeVisible();
  });

  test("no AWS credential or S3 paths in DOM text", async ({ page }) => {
    const bodyText = await page.evaluate(() => document.body.innerText);
    expect(bodyText).not.toMatch(/s3:\/\//i);
    expect(bodyText).not.toMatch(/AKIA[A-Z0-9]{16}/);
    expect(bodyText).not.toMatch(/aws_access_key_id/i);
  });

  test("no sign-in prompts on landing", async ({ page }) => {
    const bodyText = await page.evaluate(() => document.body.innerText);
    expect(bodyText).not.toMatch(/sign in to continue/i);
    expect(bodyText).not.toMatch(/administrator access required/i);
  });
});

test.describe("Workspace", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: /Explore the targets/i }).click();
  });

  test("shows sidebar with Target Prioritization brand", async ({ page }) => {
    await expect(page.getByText(/Target\s*Prioritization/i).first()).toBeVisible();
  });

  test("sidebar has public demo and admin mode buttons", async ({ page }) => {
    await expect(page.getByRole("button", { name: /Public demo/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /Admin research/i })).toBeVisible();
  });

  test("Home button returns to landing page", async ({ page }) => {
    await page.getByTitle("Back to landing").click();
    await expect(page.getByRole("heading", { name: /Drug Target Prioritization/i })).toBeVisible();
  });

  test("no AWS credentials in DOM", async ({ page }) => {
    const bodyText = await page.evaluate(() => document.body.innerText);
    expect(bodyText).not.toMatch(/AKIA[A-Z0-9]{16}/);
    expect(bodyText).not.toMatch(/aws_secret_access_key/i);
  });
});

test.describe("Theme toggle", () => {
  test("theme toggle switches dark/light on landing", async ({ page }) => {
    await page.goto("/");
    const html = page.locator("html");
    await expect(html).not.toHaveAttribute("data-theme", "dark");
    await page.getByRole("button", { name: /Dark/i }).click();
    await expect(html).toHaveAttribute("data-theme", "dark");
    await page.getByRole("button", { name: /Light/i }).click();
    await expect(html).not.toHaveAttribute("data-theme", "dark");
  });
});
