// Render the exact paired eight-window timeline evidence; no event changes.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const sharp = require('sharp');

const digest = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');

async function main() {
  const [manifestFile, destination] = process.argv.slice(2);
  const report = JSON.parse(fs.readFileSync(manifestFile, 'utf8'));
  if (report.records.length !== 8 || new Set(report.records.map(r => r.record_index)).size !== 8) {
    throw new Error('Expected eight distinct paired windows');
  }
  fs.mkdirSync(destination, {recursive: false});
  sharp.concurrency(1);
  const manifest = {source: manifestFile, source_sha256: digest(manifestFile), sheets: []};
  for (let start = 0; start < 8; start += 2) {
    const rows = report.records.slice(start, start + 2);
    const sources = rows.flatMap(row => ['control', 'candidate'].map(arm => row.variants[arm]));
    const images = [];
    for (const source of sources) {
      if (digest(source.svg) !== source.svg_sha256) throw new Error('Raw timeline changed');
      images.push(await sharp(source.svg).png().toBuffer());
    }
    const file = path.join(destination, `paired-${start / 2}.png`);
    await sharp({create: {width: 2000, height: 2040, channels: 4, background: '#101720'}})
      .composite(images.map((input, i) => ({input, left: 0, top: i * 510}))).png().toFile(file);
    manifest.sheets.push({file, sha256: digest(file), record_indices: rows.map(r => r.record_index),
      sources: sources.map(s => ({path: s.svg, sha256: s.svg_sha256}))});
  }
  fs.writeFileSync(path.join(destination, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  process.stdout.write(JSON.stringify({sheets: 4, paired_windows: 8, destination}) + '\n');
}

main().catch(error => { process.stderr.write(String(error) + '\n'); process.exitCode = 1; });
