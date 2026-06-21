import { useState, useRef, useCallback } from "react";

const style = `
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Mono:wght@300;400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0c0b09;
    --bg2: #131210;
    --bg3: #1a1814;
    --border: #2e2b26;
    --border-hover: #5a5040;
    --amber: #d97706;
    --amber-light: #f59e0b;
    --amber-dim: rgba(217, 119, 6, 0.15);
    --amber-glow: rgba(217, 119, 6, 0.08);
    --text: #e8e0d0;
    --text-dim: #7a7060;
    --text-mid: #b0a090;
    --success: #10b981;
  }

  body { background: var(--bg); color: var(--text); font-family: 'DM Mono', monospace; }

  .app {
    min-height: 100vh;
    background: var(--bg);
    background-image:
      radial-gradient(ellipse 80% 50% at 50% -20%, rgba(217,119,6,0.06) 0%, transparent 60%),
      url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
    padding: 0 0 80px;
  }

  /* Header */
  .header {
    padding: 36px 60px 28px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 16px;
    position: relative;
  }
  .header::after {
    content: '';
    position: absolute;
    bottom: -1px; left: 60px;
    width: 120px; height: 1px;
    background: var(--amber);
    opacity: 0.6;
  }
  .app-icon {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    filter: drop-shadow(0 0 8px rgba(217,119,6,0.25));
  }
  .header-text {
    display: flex;
    align-items: baseline;
    gap: 12px;
  }
  .logo {
    font-family: 'Cormorant Garamond', serif;
    font-size: 32px;
    font-weight: 300;
    letter-spacing: -0.02em;
    color: var(--text);
  }
  .logo em {
    font-style: italic;
    color: var(--amber-light);
  }
  .logo-tag {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    font-weight: 400;
    color: var(--text-dim);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 3px 8px;
    border: 1px solid var(--border);
    border-radius: 2px;
    margin-left: 4px;
    vertical-align: middle;
  }

  /* Main layout */
  .main {
    max-width: 1100px;
    margin: 0 auto;
    padding: 52px 40px 0;
  }

  .section-label {
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  /* Upload grid */
  .upload-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 32px;
  }

  .upload-zone {
    position: relative;
    border: 1px dashed var(--border);
    border-radius: 4px;
    background: var(--bg2);
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s, box-shadow 0.2s;
    overflow: hidden;
    min-height: 280px;
    display: flex;
    flex-direction: column;
  }
  .upload-zone:hover, .upload-zone.drag-over {
    border-color: var(--amber);
    background: var(--bg3);
    box-shadow: 0 0 40px var(--amber-glow), inset 0 0 40px var(--amber-glow);
  }
  .upload-zone.has-image {
    border-style: solid;
    border-color: var(--border-hover);
  }
  .upload-zone.has-image:hover {
    border-color: var(--amber);
  }

  .upload-inner {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px 32px;
    gap: 14px;
    transition: opacity 0.2s;
  }
  .upload-zone.has-image .upload-inner {
    display: none;
  }

  .upload-icon {
    width: 40px;
    height: 40px;
    opacity: 0.25;
    transition: opacity 0.2s;
  }
  .upload-zone:hover .upload-icon { opacity: 0.5; }

  .upload-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 20px;
    font-weight: 400;
    color: var(--text-mid);
    text-align: center;
  }
  .upload-sub {
    font-size: 11px;
    color: var(--text-dim);
    text-align: center;
    line-height: 1.6;
  }

  .upload-zone input[type="file"] {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
    width: 100%;
    height: 100%;
  }

  /* Image preview */
  .img-preview {
    position: absolute;
    inset: 0;
    display: none;
  }
  .upload-zone.has-image .img-preview {
    display: block;
  }
  .img-preview img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  .img-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(to top, rgba(12,11,9,0.9) 0%, transparent 50%);
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding: 14px 16px;
    gap: 4px;
  }
  .img-name {
    font-size: 11px;
    color: var(--text-mid);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .img-change {
    font-size: 10px;
    color: var(--amber);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    opacity: 0;
    transition: opacity 0.2s;
  }
  .upload-zone:hover .img-change { opacity: 1; }

  .zone-badge {
    position: absolute;
    top: 12px;
    left: 12px;
    font-size: 9px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-dim);
    background: rgba(12,11,9,0.85);
    border: 1px solid var(--border);
    padding: 3px 8px;
    border-radius: 2px;
    z-index: 10;
  }

  /* Arrow between zones */
  .zone-arrow {
    display: flex;
    align-items: center;
    justify-content: center;
    align-self: center;
  }
  .zone-arrow svg {
    color: var(--text-dim);
    opacity: 0.3;
  }

  /* Stylize button */
  .action-row {
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 48px;
  }

  .stylize-btn {
    position: relative;
    padding: 14px 40px;
    background: var(--amber);
    color: #0c0b09;
    border: none;
    border-radius: 2px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    cursor: pointer;
    transition: background 0.15s, transform 0.1s, box-shadow 0.15s;
    overflow: hidden;
  }
  .stylize-btn:hover:not(:disabled) {
    background: var(--amber-light);
    box-shadow: 0 0 30px rgba(217,119,6,0.4);
    transform: translateY(-1px);
  }
  .stylize-btn:active:not(:disabled) {
    transform: translateY(0);
  }
  .stylize-btn:disabled {
    background: var(--bg3);
    color: var(--text-dim);
    cursor: not-allowed;
    border: 1px solid var(--border);
  }
  .stylize-btn.loading {
    background: var(--bg3);
    color: var(--amber);
    border: 1px solid var(--amber);
    cursor: not-allowed;
  }

  .btn-shimmer {
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.15) 50%, transparent 100%);
    transform: translateX(-100%);
    animation: shimmer 1.8s infinite;
  }
  @keyframes shimmer {
    to { transform: translateX(200%); }
  }

  .action-hint {
    font-size: 11px;
    color: var(--text-dim);
    line-height: 1.6;
  }
  .action-hint strong {
    color: var(--text-mid);
    font-weight: 400;
  }

  /* Status bar */
  .status-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 16px;
    border: 1px solid var(--border);
    border-radius: 2px;
    background: var(--bg2);
    font-size: 11px;
    color: var(--text-dim);
    margin-bottom: 12px;
    animation: fadeIn 0.3s ease;
  }
  .status-bar.error {
    border-color: rgba(239,68,68,0.3);
    color: #f87171;
    background: rgba(239,68,68,0.05);
  }
  .status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--amber);
    animation: pulse 1.2s ease-in-out infinite;
    flex-shrink: 0;
  }
  .status-bar.error .status-dot {
    background: #f87171;
    animation: none;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.7); }
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(-6px); }
    to { opacity: 1; transform: translateY(0); }
  }

  /* Result */
  .result-section {
    animation: revealUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
  }
  @keyframes revealUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .result-card {
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    background: var(--bg2);
    position: relative;
  }
  .result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--amber), transparent);
    opacity: 0.5;
  }

  .result-img {
    width: 100%;
    display: block;
    max-height: 600px;
    object-fit: contain;
    background: #080807;
  }

  .result-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px;
    border-top: 1px solid var(--border);
  }
  .result-meta {
    font-size: 11px;
    color: var(--text-dim);
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .result-meta span {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .meta-dot {
    width: 4px; height: 4px;
    border-radius: 50%;
    background: var(--success);
  }

  .download-btn {
    padding: 8px 20px;
    background: transparent;
    color: var(--amber);
    border: 1px solid var(--amber);
    border-radius: 2px;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    cursor: pointer;
    text-decoration: none;
    transition: background 0.15s, box-shadow 0.15s;
    display: inline-block;
  }
  .download-btn:hover {
    background: var(--amber-dim);
    box-shadow: 0 0 20px rgba(217,119,6,0.2);
  }

  /* Thumbnail strip */
  .thumb-strip {
    display: flex;
    gap: 8px;
    padding: 16px 20px;
    border-top: 1px solid var(--border);
    background: var(--bg);
  }
  .thumb {
    width: 60px;
    height: 60px;
    border-radius: 2px;
    overflow: hidden;
    border: 1px solid var(--border);
    flex-shrink: 0;
    position: relative;
  }
  .thumb img { width: 100%; height: 100%; object-fit: cover; }
  .thumb-label {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    font-size: 8px;
    text-align: center;
    padding: 2px;
    background: rgba(12,11,9,0.8);
    color: var(--text-dim);
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }
  .thumb-arrow {
    display: flex;
    align-items: center;
    color: var(--text-dim);
    opacity: 0.3;
    font-size: 16px;
  }

  /* Responsive */
  @media (max-width: 640px) {
    .header { padding: 24px 20px; }
    .main { padding: 32px 16px 0; }
    .upload-grid { grid-template-columns: 1fr; }
    .zone-arrow { display: none; }
  }
`;

function AppIcon() {
  return (
    <svg
      width="42" height="42"
      viewBox="0 0 42 42"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-label="Neural Style Transfer icon"
    >
      {/* Outer frame */}
      <rect x="1" y="1" width="40" height="40" rx="6"
        stroke="#d97706" strokeWidth="1.4"
        fill="rgba(217,119,6,0.07)"
      />
      {/* Inner frame subtle inset */}
      <rect x="4" y="4" width="34" height="34" rx="4"
        stroke="#d97706" strokeWidth="0.4" strokeDasharray="3 3"
        fill="none" opacity="0.25"
      />

      {/* ── LEFT SIDE: Content / photo ── */}
      {/* Sun */}
      <circle cx="10" cy="11" r="3.8"
        stroke="#d97706" strokeWidth="1.2"
        fill="rgba(217,119,6,0.12)"
      />
      {/* Sun rays (4 tiny lines) */}
      <line x1="10" y1="5.8" x2="10" y2="4.5"   stroke="#d97706" strokeWidth="0.9" strokeLinecap="round" opacity="0.55"/>
      <line x1="10" y1="16.2" x2="10" y2="17.5" stroke="#d97706" strokeWidth="0.9" strokeLinecap="round" opacity="0.55"/>
      <line x1="4.8" y1="11" x2="3.5" y2="11"   stroke="#d97706" strokeWidth="0.9" strokeLinecap="round" opacity="0.55"/>
      <line x1="15.2" y1="11" x2="16.5" y2="11" stroke="#d97706" strokeWidth="0.9" strokeLinecap="round" opacity="0.55"/>
      {/* Mountain silhouette */}
      <path d="M5 33 L10 22 L15 28 L18 24"
        stroke="#d97706" strokeWidth="1.3"
        strokeLinecap="round" strokeLinejoin="round"
        fill="none" opacity="0.85"
      />
      {/* Ground line */}
      <line x1="5" y1="35" x2="18" y2="35"
        stroke="#d97706" strokeWidth="0.9"
        strokeLinecap="round" opacity="0.35"
      />
      {/* Horizon hint */}
      <line x1="5" y1="29.5" x2="18" y2="29.5"
        stroke="#d97706" strokeWidth="0.5"
        strokeLinecap="round" opacity="0.2"
      />

      {/* ── CENTER: Neural nodes ── */}
      <circle cx="21" cy="13" r="2.2" fill="#d97706"/>
      <circle cx="21" cy="21" r="2.2" fill="#d97706"/>
      <circle cx="21" cy="29" r="2.2" fill="#d97706"/>
      {/* Vertical connections */}
      <line x1="21" y1="15.2" x2="21" y2="18.8"
        stroke="#d97706" strokeWidth="0.8" opacity="0.45"
      />
      <line x1="21" y1="23.2" x2="21" y2="26.8"
        stroke="#d97706" strokeWidth="0.8" opacity="0.45"
      />
      {/* Left feed lines */}
      <line x1="18" y1="13" x2="19.8" y2="13"
        stroke="#d97706" strokeWidth="0.7" opacity="0.4"
      />
      <line x1="18" y1="21" x2="19.8" y2="21"
        stroke="#d97706" strokeWidth="0.7" opacity="0.4"
      />
      <line x1="18" y1="29" x2="19.8" y2="29"
        stroke="#d97706" strokeWidth="0.7" opacity="0.4"
      />
      {/* Right feed lines */}
      <line x1="22.2" y1="13" x2="24" y2="13"
        stroke="#f59e0b" strokeWidth="0.7" opacity="0.4"
      />
      <line x1="22.2" y1="21" x2="24" y2="21"
        stroke="#f59e0b" strokeWidth="0.7" opacity="0.4"
      />
      <line x1="22.2" y1="29" x2="24" y2="29"
        stroke="#f59e0b" strokeWidth="0.7" opacity="0.4"
      />

      {/* ── RIGHT SIDE: Style / brush strokes ── */}
      <path d="M24 12 C26 10, 29 11.5, 32 10 C33.5 9.3, 35 10.2, 37 9.5"
        stroke="#f59e0b" strokeWidth="1.5"
        strokeLinecap="round" fill="none"
      />
      <path d="M24 17 C26.5 15.5, 30 16.5, 33 15 C34.5 14.3, 36 15.5, 37 14.5"
        stroke="#f59e0b" strokeWidth="1.5"
        strokeLinecap="round" fill="none" opacity="0.8"
      />
      <path d="M24 21 C26 19.5, 29 20.5, 32 19 C34 18, 35.5 19.5, 37 18.5"
        stroke="#f59e0b" strokeWidth="1.5"
        strokeLinecap="round" fill="none"
      />
      <path d="M24 26 C27 24.5, 30 25.5, 33 24 C35 23, 36 24.5, 37 23.5"
        stroke="#f59e0b" strokeWidth="1.3"
        strokeLinecap="round" fill="none" opacity="0.7"
      />
      <path d="M24 31 C26.5 29.5, 30 30.5, 33 29 C34.5 28.2, 36 29.5, 37 28.5"
        stroke="#f59e0b" strokeWidth="1.1"
        strokeLinecap="round" fill="none" opacity="0.5"
      />
    </svg>
  );
}


function UploadZone({ label, image, onImage }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const toBase64 = (file) =>
    new Promise((res) => {
      const r = new FileReader();
      r.onload = () => res(r.result);
      r.readAsDataURL(file);
    });

  const handle = useCallback(
    async (file) => {
      if (!file || !file.type.startsWith("image/")) return;
      const b64 = await toBase64(file);
      onImage({ src: b64, name: file.name, size: file.size });
    },
    [onImage]
  );

  return (
    <div
      className={`upload-zone${image ? " has-image" : ""}${dragging ? " drag-over" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        handle(e.dataTransfer.files[0]);
      }}
    >
      <span className="zone-badge">{label}</span>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={(e) => handle(e.target.files[0])}
      />

      <div className="upload-inner">
        <svg className="upload-icon" viewBox="0 0 40 40" fill="none" stroke="currentColor" strokeWidth="1.2">
          <rect x="4" y="8" width="32" height="24" rx="2" />
          <circle cx="14" cy="18" r="3" />
          <path d="M4 28l9-8 7 6 5-4 11 10" />
        </svg>
        <div className="upload-title">Drop {label.toLowerCase()} here</div>
        <div className="upload-sub">
          or click to browse<br />
          <span style={{ opacity: 0.5 }}>PNG, JPG, WEBP</span>
        </div>
      </div>

      {image && (
        <div className="img-preview">
          <img src={image.src} alt={label} />
          <div className="img-overlay">
            <div className="img-name">{image.name}</div>
            <div className="img-change">↑ Change image</div>
          </div>
        </div>
      )}
    </div>
  );
}

function formatBytes(b) {
  if (b < 1024) return b + " B";
  if (b < 1024 * 1024) return (b / 1024).toFixed(1) + " KB";
  return (b / (1024 * 1024)).toFixed(2) + " MB";
}

export default function StyleTransfer() {
  const [content, setContent] = useState(null);
  const [styleImg, setStyleImg] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [elapsed, setElapsed] = useState(null);

  const canRun = content && styleImg && !loading;

  const runStylize = async () => {
    if (!canRun) return;
    setLoading(true);
    setError(null);
    setResult(null);
    const t0 = Date.now();

    try {
      const stripPrefix = (dataUrl) => {
        const idx = dataUrl.indexOf(",");
        return idx !== -1 ? dataUrl.substring(idx + 1) : dataUrl;
      };

      const resp = await fetch("/api/style-transfer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content_img: stripPrefix(content.src),
          style_img: stripPrefix(styleImg.src),
        }),
      });

      if (!resp.ok) throw new Error(`Server returned ${resp.status}: ${resp.statusText}`);

      const json = await resp.json();
      const raw = json.image;
      if (!raw) throw new Error("Could not find image field in JSON response");
      const imgSrc = `data:image/png;base64,${raw}`;

      setElapsed(((Date.now() - t0) / 1000).toFixed(1));
      setResult(imgSrc);
    } catch (err) {
      setError(err.message || "Unknown error — is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const downloadResult = () => {
    const a = document.createElement("a");
    a.href = result;
    a.download = "stylized_output.png";
    a.click();
  };

  return (
    <>
      <style>{style}</style>
      <div className="app">
        <header className="header">
          <div className="app-icon">
            <AppIcon />
          </div>
          <div className="header-text">
            <h1 className="logo">
              <em>Neural</em> Style Transfer
            </h1>
            <span className="logo-tag">v1.0</span>
          </div>
        </header>

        <main className="main">
          <div className="section-label">01 — Upload images</div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr", gap: "20px", alignItems: "stretch", marginBottom: "28px" }}>
            <UploadZone label="Content" image={content} onImage={setContent} />
            <div className="zone-arrow">
              <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                <path d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" />
              </svg>
            </div>
            <UploadZone label="Style" image={styleImg} onImage={setStyleImg} />
          </div>

          <div className="section-label">02 — Run transfer</div>

          {error && (
            <div className="status-bar error">
              <div className="status-dot" />
              {error}
            </div>
          )}

          {loading && (
            <div className="status-bar">
              <div className="status-dot" />
              Transferring style… this may take a moment
            </div>
          )}

          <div className="action-row">
            <button
              className={`stylize-btn${loading ? " loading" : ""}`}
              onClick={runStylize}
              disabled={!canRun}
            >
              {loading && <div className="btn-shimmer" />}
              {loading ? "Synthesizing…" : "Stylize image"}
            </button>
            <div className="action-hint">
              {!content && !styleImg && <>Upload a <strong>content</strong> and <strong>style</strong> image to begin</>}
              {content && !styleImg && <>Now upload a <strong>style</strong> image</>}
              {!content && styleImg && <>Now upload a <strong>content</strong> image</>}
              {content && styleImg && !loading && <>Ready — click to transfer style to your image</>}
              {loading && <>Calling <strong>localhost:3000/stylize</strong></>}
            </div>
          </div>

          {result && (
            <>
              <div className="section-label">03 — Result</div>
              <div className="result-section">
                <div className="result-card">
                  <img className="result-img" src={result} alt="Stylized output" />

                  {content && styleImg && (
                    <div className="thumb-strip">
                      <div className="thumb">
                        <img src={content.src} alt="content" />
                        <div className="thumb-label">Content</div>
                      </div>
                      <div className="thumb-arrow">+</div>
                      <div className="thumb">
                        <img src={styleImg.src} alt="style" />
                        <div className="thumb-label">Style</div>
                      </div>
                      <div className="thumb-arrow">→</div>
                      <div className="thumb">
                        <img src={result} alt="result" />
                        <div className="thumb-label">Output</div>
                      </div>
                    </div>
                  )}

                  <div className="result-footer">
                    <div className="result-meta">
                      <span>
                        <div className="meta-dot" />
                        Transfer complete
                      </span>
                      {elapsed && <span style={{ opacity: 0.5 }}>{elapsed}s</span>}
                    </div>
                    <button className="download-btn" onClick={downloadResult}>
                      ↓ Download
                    </button>
                  </div>
                </div>
              </div>
            </>
          )}
        </main>
      </div>
    </>
  );
}