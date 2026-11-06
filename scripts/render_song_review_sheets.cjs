// Render every saved full-song and detail SVG; never edit the source events.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

async function main() {
  const [reportPath, outputPath, sharpPath] = process.argv.slice(2);
  if (!reportPath || !outputPath || !sharpPath) {
    throw new Error('Expected review-report, new output-directory, and sharp module path');
  }
  const sharp = require(path.resolve(sharpPath));
  const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
  if (report.records.length !== 20 || new Set(report.records.map(r => r.chart_id)).size !== 20) {
    throw new Error('Need all twenty distinct song attempts, including failures');
  }
  if (fs.existsSync(outputPath)) throw new Error('Inspection output already exists');
  const sha = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
  // Verify all source figures before creating the output directory.
  const kinds = ['comparison', 'active-detail'];
  const sources = kinds.flatMap(kind => report.records.map(r =>
    path.join(path.dirname(r.preview), `${kind}.svg`)));
  const sourceSha = Object.fromEntries([reportPath, __filename, ...sources].map(p => [p, sha(p)]));
  fs.mkdirSync(outputPath, { recursive: true });
  const manifest = { review: reportPath, source_sha256: sourceSha, sheets: [],
    actual_visual_review_complete: false, audio_audition_complete: false };
  for (const kind of kinds) {
    for (let start = 0; start < report.records.length; start += 4) {
      const rows = report.records.slice(start, start + 4);
      const files = rows.map(r => path.join(path.dirname(r.preview), `${kind}.svg`));
      const buffers = await Promise.all(files.map(file => sharp(file).resize(1200).png().toBuffer()));
      const metadata = await Promise.all(buffers.map(buffer => sharp(buffer).metadata()));
      let height = 0;
      const overlays = buffers.map((input, i) => {
        const layer = { input, top: height, left: 0 };
        height += metadata[i].height;
        return layer;
      });
      const file = path.join(outputPath, `${kind}-${start / 4}.png`);
      await sharp({ create: { width: 1200, height, channels: 4, background: '#101720' } })
        .composite(overlays).png().toFile(file);
      manifest.sheets.push({ file, sha256: sha(file), sources: files,
        chart_ids: rows.map(r => r.chart_id), kind });
    }
  }
  fs.writeFileSync(path.join(outputPath, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  console.log(JSON.stringify({ songs: 20, sheets: manifest.sheets.length, output: outputPath }));
}

main().catch(error => { console.error(error); process.exitCode = 1; });
