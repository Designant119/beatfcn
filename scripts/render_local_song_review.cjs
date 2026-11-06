// Rasterize the exact raw-event SVG evidence into four-song review sheets.
// NODE_PATH may point to the desktop's bundled dependency directory.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const sharp = require('sharp');

async function main() {
  const [reportFile, destination] = process.argv.slice(2);
  if (!reportFile || !destination) throw new Error('Expected report.json and new inspection directory');
  const report = JSON.parse(fs.readFileSync(reportFile, 'utf8'));
  if (report.records.length !== 20 || new Set(report.records.map(r => r.chart_id)).size !== 20) {
    throw new Error('Expected twenty distinct actual song artifacts');
  }
  fs.mkdirSync(destination, {recursive: false});
  sharp.concurrency(1);
  const manifest = {report: reportFile, report_sha256: digest(reportFile), sheets: []};
  for (const kind of ['comparison', 'active-detail']) {
    for (let start = 0; start < 20; start += 4) {
      const records = report.records.slice(start, start + 4);
      const sources = records.map(r => path.join(path.dirname(r.preview), `${kind}.svg`));
      const images = [];
      for (const source of sources) images.push(await sharp(source).png().toBuffer());
      const file = path.join(destination, `${kind}-${start/4}.png`);
      await sharp({create: {width: 2000, height: 2040, channels: 4, background: '#101720'}})
        .composite(images.map((input, i) => ({input, left: 0, top: i*510}))).png().toFile(file);
      manifest.sheets.push({file, sha256: digest(file), chart_ids: records.map(r => r.chart_id),
        sources: sources.map(source => ({path: source, sha256: digest(source)}))});
    }
  }
  fs.writeFileSync(path.join(destination, 'manifest.json'), JSON.stringify(manifest, null, 2)+'\n');
  process.stdout.write(JSON.stringify({sheets: manifest.sheets.length, songs: 20, destination})+'\n');
}

function digest(file) { return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex'); }
main().catch(error => { process.stderr.write(String(error)+'\n'); process.exitCode = 1; });
