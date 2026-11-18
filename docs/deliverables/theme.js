// Writes the deck's colour scheme into ppt/theme/theme1.xml.
// pptxgenjs sets theme fonts but always writes Office's default palette, so scheme colours
// (C.accent1 ...) would otherwise resolve to Office blue/orange.
const fs = require("fs");
const JSZip = require("jszip");

const SLOTS = ["dk1", "lt1", "dk2", "lt2", "accent1", "accent2", "accent3", "accent4", "accent5", "accent6", "hlink", "folHlink"];
const escapeXml = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

async function applyTheme(file, theme) {
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const part = "ppt/theme/theme1.xml";
  let xml = await zip.file(part).async("string");
  const scheme = SLOTS.map((slot) => {
    const hex = theme.colors[slot];
    if (!/^[0-9A-Fa-f]{6}$/.test(hex || "")) throw new Error(`theme.colors.${slot} must be 6-digit hex, got ${hex}`);
    return `<a:${slot}><a:srgbClr val="${hex.toUpperCase()}"/></a:${slot}>`;
  }).join("");
  const replaced = xml.replace(/<a:clrScheme\b[^>]*>[\s\S]*?<\/a:clrScheme>/,
    `<a:clrScheme name="${escapeXml(theme.name)}">${scheme}</a:clrScheme>`);
  if (replaced === xml) throw new Error("No <a:clrScheme> found in theme1.xml");
  xml = replaced.replace(/(<a:theme\b[^>]*\bname=")[^"]*(")/, `$1${escapeXml(theme.name)}$2`);
  zip.file(part, xml);
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

module.exports = { applyTheme };
