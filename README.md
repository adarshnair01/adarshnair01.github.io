# adarshnair.online

Personal portfolio of **Adarsh Nair**, Data Science Manager & ML Engineer.
Live at **https://adarshnair.online**.

Built with React 19, Vite, Tailwind CSS and Framer Motion. The current look is a
Christmas & New Year theme: falling snow, fairy lights, a New Year countdown, and a
few small games.

## Run locally

```bash
npm install
npm run dev      # start the dev server
npm run lint     # check the code
npm run build    # production build into dist/
```

## Deploying

Every push to `master` runs `.github/workflows/deploy.yml`, which lints, builds and
publishes `dist/` to the `gh-pages` branch (GitHub Pages, custom domain via `public/CNAME`).

## Where things live

| What | Where |
|---|---|
| Links (GitHub, LinkedIn, email, résumé) | `src/siteConfig.js` |
| Colours and fonts | `tailwind.config.js`, `src/index.css`, `index.html` (font link) |
| Shared decorations (snow, lights, baubles, countdown) | `src/components/Festive.jsx` |
| Page sections | `src/components/` (`Hero`, `About`, `Experience`, `Projects`, `QuickGames`, `Footer`) |
| Mini-games | `src/components/games/` |
| SEO, link-preview tags, structured data | `index.html`, `public/og-image.jpg`, `public/sitemap.xml` |

### Updating the résumé link

`RESUME_URL` in `src/siteConfig.js` currently points to LinkedIn. To link a PDF instead,
add the file to `public/` and set `RESUME_URL` to its path, e.g. `'/Adarsh_Nair_Resume.pdf'`.
