# MedAssist web interface

React 19 + Vite + TypeScript. Design system: [../DESIGN.md](../DESIGN.md); product record: [../PRODUCT.md](../PRODUCT.md).

```bash
npm install
npm run dev     # http://127.0.0.1:5173, proxies API calls to the backend on :8000
npm run build   # writes dist/, which FastAPI serves at http://127.0.0.1:8000
```

- `src/api.ts`: typed API client and the server-sent-events reader for streamed answers
- `src/leader.tsx`: leader lines (the evidence-tracing signature interaction)
- `src/components/`: case band, image plate, note panel, evidence column, conversation, new case
- `src/styles.css`: all design tokens and component styles
