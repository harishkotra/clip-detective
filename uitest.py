import asyncio, os, sys
from playwright.async_api import async_playwright

ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(ROOT, "test", "bunny.mp4")
SHOT = os.path.join(ROOT, "shots", "ui.png")

async def main():
    os.makedirs(os.path.join(ROOT, "shots"), exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome", args=["--no-proxy-server"])
        pg = await b.new_page(viewport={"width": 1400, "height": 1000})
        pg.on("console", lambda m: print("console:", m.text[:200]))
        pg.on("pageerror", lambda e: print("pageerror:", str(e)[:300]))
        await pg.goto("http://127.0.0.1:8787/", wait_until="load")
        await pg.set_input_files("#file", VIDEO)
        await pg.wait_for_selector(".frame", timeout=120000)
        n = await pg.locator(".frame").count()
        print("frames in strip:", n)
        await pg.click(".chip >> nth=0")
        await pg.wait_for_selector(".a", timeout=300000)
        print("ANSWER:", (await pg.inner_text(".a"))[:400])
        print("META:", await pg.inner_text(".log .meta >> nth=-1"))
        await pg.wait_for_timeout(1500)
        await pg.screenshot(path=SHOT, full_page=True)
        print("shot:", SHOT)
        await b.close()

asyncio.run(main())
