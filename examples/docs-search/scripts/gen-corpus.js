// gen-corpus.js — one-shot offline script that turns react.dev's MDX docs into
// a flat chunks.json for the demo.
//
// Run once: `node examples/docs-search/scripts/gen-corpus.js <path-to-react.dev>`
// Output:   examples/docs-search/chunks.json
//
// What it does:
//   1. Walks <react.dev>/src/content/{learn,reference}/**/*.md
//   2. Parses frontmatter (title), tracks the section heading stack
//   3. Strips MDX/JSX components we can't render meaningfully:
//      - <Sandpack>, <Square>, <ChatRoom>, <Profile>, ... (interactive examples)
//      - <Diagram>, <Heading>, <Item> (visual/structural)
//      - <YouWillLearn>, <Solution>, <Hint> (preview lists, exercise UI)
//      But KEEPS the prose contents of:
//      - <Intro>      (page summaries — highly searchable)
//      - <Note>       (callouts)
//      - <DeepDive>   (deeper explanations)
//      - <Recap>      (page recaps)
//   4. Skips code fences (```...```).
//   5. Emits one chunk per paragraph, with the title+heading prepended for
//      semantic context (so the embedding "knows" what section it's from).
//
// The output is a static asset shipped with the demo. Visitors don't pull
// from react.dev at runtime.

const fs = require('node:fs');
const path = require('node:path');

// ── Args ────────────────────────────────────────────────────────────────────

const sourceRoot = process.argv[2];
if (!sourceRoot || !fs.existsSync(sourceRoot)) {
    console.error('Usage: node gen-corpus.js <path-to-react.dev>');
    console.error('Example: node gen-corpus.js /path/to/react.dev');
    process.exit(1);
}

const CONTENT_DIR = path.join(sourceRoot, 'src', 'content');
const OUT_PATH = path.join(__dirname, '..', 'chunks.json');

const REACT_DEV_BASE = 'https://react.dev';

// JSX tags whose inner prose we keep — they're real content, just wrapped.
const KEEP_INNER = new Set(['Intro', 'Note', 'DeepDive', 'Recap', 'Wip']);

// JSX tags whose inner contents we drop entirely (interactive examples,
// exercise UI, visual elements that don't have embeddable text).
const DROP_TAGS = new Set([
    'Sandpack', 'Square', 'ChatRoom', 'Profile', 'Diagram', 'Heading',
    'Item', 'YouWillLearn', 'Solution', 'Hint', 'Section', 'Trans',
    'Challenges', 'Pitfall', 'Illustration', 'IllustrationBlock',
    'TerminalBlock', 'Math', 'MathI', 'IntlProvider', 'TeamMember',
    'YouTubeIframe', 'ConferenceLayout', 'CommunityImages',
    'TextBlock', 'ConsoleBlock', 'ConsoleLogLine', 'TwitterContributors',
    'NewsletterForm', 'TeamMembers', 'ChallengesItem',
]);

// ── File discovery ─────────────────────────────────────────────────────────

function walkMarkdown(dir, out = []) {
    if (!fs.existsSync(dir)) return out;
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            walkMarkdown(full, out);
        } else if (entry.isFile() && entry.name.endsWith('.md')) {
            out.push(full);
        }
    }
    return out;
}

// ── MDX cleaning ───────────────────────────────────────────────────────────

// Parse the YAML-ish frontmatter at the top of an MDX file.
// Only handles single-line `key: "value"` entries — that's all react.dev uses.
function parseFrontmatter(raw) {
    if (!raw.startsWith('---\n')) return { meta: {}, body: raw };
    const end = raw.indexOf('\n---\n', 4);
    if (end < 0) return { meta: {}, body: raw };
    const fmText = raw.slice(4, end);
    const body = raw.slice(end + 5);
    const meta = {};
    for (const line of fmText.split('\n')) {
        const m = line.match(/^([a-zA-Z_]+):\s*"?(.+?)"?\s*$/);
        if (m) meta[m[1]] = m[2];
    }
    return { meta, body };
}

// Drop the contents of any opening/closing JSX tag pairs in DROP_TAGS, and
// unwrap KEEP_INNER tags (keeping their inner content). Self-closing tags
// (e.g. `<Sandpack />`) are dropped as well.
function stripJsx(body) {
    let out = body;

    // First strip drop-tag content (paired form: <Tag>...</Tag>)
    for (const tag of DROP_TAGS) {
        const re = new RegExp(`<${tag}\\b[^>]*>[\\s\\S]*?</${tag}>`, 'g');
        out = out.replace(re, '');
    }
    // Self-closing form: <Tag ... />
    for (const tag of DROP_TAGS) {
        const re = new RegExp(`<${tag}\\b[^>]*/>`, 'g');
        out = out.replace(re, '');
    }
    // Unwrap keep-inner tags (paired form): keep contents, drop tag itself
    for (const tag of KEEP_INNER) {
        const re = new RegExp(`<${tag}\\b[^>]*>([\\s\\S]*?)</${tag}>`, 'g');
        out = out.replace(re, '$1');
    }
    // Self-closing keep tags (rare) — just drop
    for (const tag of KEEP_INNER) {
        const re = new RegExp(`<${tag}\\b[^>]*/>`, 'g');
        out = out.replace(re, '');
    }

    // Any remaining unknown JSX tag pairs — unwrap (keep inner text)
    out = out.replace(/<([A-Z][a-zA-Z]*)\b[^>]*>([\s\S]*?)<\/\1>/g, '$2');
    // Drop any remaining self-closing custom JSX tags
    out = out.replace(/<[A-Z][a-zA-Z]*\b[^>]*\/>/g, '');

    return out;
}

// Strip standard markdown link syntax to plain text: [text](url) → text
function stripMarkdown(text) {
    return text
        .replace(/`([^`]+)`/g, '$1')                           // inline code
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')              // links
        .replace(/!\[[^\]]*\]\([^)]+\)/g, '')                  // images
        .replace(/^\s*[-*]\s+/gm, '')                          // list bullets
        .replace(/^\s*\d+\.\s+/gm, '')                         // numbered lists
        .replace(/\*\*([^*]+)\*\*/g, '$1')                     // bold
        .replace(/\*([^*]+)\*/g, '$1')                         // italic
        .replace(/_([^_]+)_/g, '$1')                           // italic alt
        .replace(/\s+/g, ' ')                                  // collapse whitespace
        .trim();
}

// Walk the body line-by-line:
//   - track the heading stack (h1 → h2 → h3) and react.dev's custom heading ID syntax `## Title {/*id*/}`
//   - skip code fences (```)
//   - skip HTML comments
//   - collect contiguous prose lines as a paragraph
function* iterateChunks(filePath, body, fileTitle) {
    const headingStack = [];   // [{ level, text, id }]
    let inCodeFence = false;
    let buffer = [];

    const flush = function* () {
        if (buffer.length === 0) return;
        const para = stripMarkdown(buffer.join(' '));
        buffer = [];
        if (!para || para.length < 25) return;   // ignore tiny fragments
        const section = headingStack.map(h => h.text).join(' · ');
        yield { fileTitle, section, headingStack: [...headingStack], paragraph: para };
    };

    for (const rawLine of body.split('\n')) {
        const line = rawLine;

        // Code fence toggle
        if (/^\s*```/.test(line)) {
            yield* flush();
            inCodeFence = !inCodeFence;
            continue;
        }
        if (inCodeFence) continue;

        // HTML comment — skip
        if (/^\s*<!--/.test(line)) continue;

        // Heading: `## Heading text {/*optional-id*/}`
        const headingMatch = line.match(/^(#{1,6})\s+(.+?)\s*(\{\/\*([^*]+)\*\/\})?\s*$/);
        if (headingMatch) {
            yield* flush();
            const level = headingMatch[1].length;
            const text = stripMarkdown(headingMatch[2]);
            const id = headingMatch[4] || null;
            while (headingStack.length && headingStack[headingStack.length - 1].level >= level) {
                headingStack.pop();
            }
            headingStack.push({ level, text, id });
            continue;
        }

        // Blank line — paragraph boundary
        if (/^\s*$/.test(line)) {
            yield* flush();
            continue;
        }

        // Plain prose line — accumulate
        buffer.push(line.trim());
    }
    yield* flush();
}

// ── URL + breadcrumb derivation ────────────────────────────────────────────

// e.g.
//   /content/learn/state-a-components-memory.md
//     → url: https://react.dev/learn/state-a-components-memory
//     → topSection: "Learn"
//   /content/reference/react/useState.md
//     → url: https://react.dev/reference/react/useState
//     → topSection: "Reference"
//   /content/learn/managing-state/index.md
//     → url: https://react.dev/learn/managing-state
function deriveUrlAndSection(filePath) {
    const rel = path.relative(CONTENT_DIR, filePath);
    let slug = rel.replace(/\\/g, '/').replace(/\.md$/, '');
    if (slug.endsWith('/index')) slug = slug.slice(0, -'/index'.length);
    const topSegment = slug.split('/')[0];
    const topSection = topSegment.charAt(0).toUpperCase() + topSegment.slice(1);
    return {
        url: `${REACT_DEV_BASE}/${slug}`,
        topSection,
        slug,
    };
}

// ── Build ──────────────────────────────────────────────────────────────────

// We intentionally include only the Learn section (conceptual "how do I X"
// content) — that's where ternlight's semantic > keyword win is biggest, and
// it caps the corpus at ~2000 chunks (~4s embed time on a modern CPU).
// Reference docs are more keyword-friendly (people search exact API names)
// and would push the corpus to 5000+ chunks.
const SECTIONS = ['learn'];

const targets = [];
for (const sub of SECTIONS) {
    targets.push(...walkMarkdown(path.join(CONTENT_DIR, sub)));
}
console.log(`Found ${targets.length} .md files across ${SECTIONS.join(', ')}/`);

const chunks = [];
let nextId = 1;
let skippedShort = 0;

for (const filePath of targets) {
    const raw = fs.readFileSync(filePath, 'utf8');
    const { meta, body } = parseFrontmatter(raw);
    const fileTitle = meta.title || path.basename(filePath, '.md');

    const cleaned = stripJsx(body);
    const { url, topSection } = deriveUrlAndSection(filePath);

    for (const ch of iterateChunks(filePath, cleaned, fileTitle)) {
        if (ch.paragraph.length < 40) { skippedShort++; continue; }

        // Build a human-readable breadcrumb: "Learn · Page Title · h2 · h3"
        const breadcrumb = [
            topSection,
            ch.fileTitle,
            ...ch.headingStack.map(h => h.text),
        ].filter(Boolean).join(' · ');

        // Build an anchor URL when we know the deepest heading's ID
        const deepest = ch.headingStack[ch.headingStack.length - 1];
        const fullUrl = deepest && deepest.id ? `${url}#${deepest.id}` : url;

        // The embedded text — prepend the breadcrumb so the embedding has
        // semantic context beyond the bare paragraph. This is the key trick
        // that makes "useState" queries match paragraphs that don't say
        // "useState" but live inside a useState section.
        const text = `${breadcrumb}. ${ch.paragraph}`;

        chunks.push({
            id:         nextId++,
            title:      ch.fileTitle,
            breadcrumb,
            section:    ch.section || null,
            url:        fullUrl,
            text,
        });
    }
}

console.log(`Skipped ${skippedShort} fragments shorter than 40 chars`);

// Write
fs.writeFileSync(OUT_PATH, JSON.stringify(chunks));
const sizeMb = (fs.statSync(OUT_PATH).size / 1024 / 1024).toFixed(2);
console.log(`Wrote ${chunks.length} chunks to ${OUT_PATH} (${sizeMb} MB raw)`);

// Quick distribution summary
const byTop = chunks.reduce((acc, c) => {
    const top = c.breadcrumb.split(' · ')[0];
    acc[top] = (acc[top] || 0) + 1;
    return acc;
}, {});
console.log('By section:');
for (const [k, v] of Object.entries(byTop)) {
    console.log(`  ${k.padEnd(15)} ${v}`);
}

// Sample of long-paragraph warning (these get truncated at 128 tokens during
// embedding — worth knowing how many)
const longChunks = chunks.filter(c => c.text.length > 700).length;
console.log(`Chunks over 700 chars (likely truncated at embed time): ${longChunks}`);
