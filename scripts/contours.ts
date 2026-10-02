// Draws a topographic contour banner of a real place, rotating daily.
//
//   node scripts/contours.ts                  # today's place
//   node scripts/contours.ts --place fuji     # a specific place
//
// Elevation comes from the public AWS Terrain Tiles (Terrarium encoding).
// Writes assets/contours-light.svg and assets/contours-dark.svg.

import { mkdirSync, writeFileSync } from "node:fs";
import { inflateSync } from "node:zlib";

type Place = { slug: string; name: string; lat: number; lon: number; km: number };

// km is the width of the strip; its height follows from the banner ratio.
const PLACES: Place[] = [
  { slug: "fuji", name: "Mount Fuji", lat: 35.3606, lon: 138.7274, km: 36 },
  { slug: "matterhorn", name: "Matterhorn", lat: 45.9763, lon: 7.6586, km: 14 },
  { slug: "snowdon", name: "Yr Wyddfa", lat: 53.0685, lon: -4.0763, km: 12 },
  { slug: "ben-nevis", name: "Ben Nevis", lat: 56.7969, lon: -5.0036, km: 14 },
  { slug: "scafell", name: "Scafell Pike", lat: 54.4542, lon: -3.2115, km: 12 },
  { slug: "grand-canyon", name: "Grand Canyon", lat: 36.0544, lon: -112.1401, km: 30 },
  { slug: "half-dome", name: "Half Dome", lat: 37.7459, lon: -119.5332, km: 12 },
  { slug: "st-helens", name: "Mount St. Helens", lat: 46.1914, lon: -122.1956, km: 16 },
  { slug: "taranaki", name: "Taranaki Maunga", lat: -39.2963, lon: 174.0634, km: 30 },
  { slug: "kilimanjaro", name: "Kilimanjaro", lat: -3.0674, lon: 37.3556, km: 50 },
  { slug: "table-mountain", name: "Table Mountain", lat: -33.9628, lon: 18.4098, km: 14 },
  { slug: "everest", name: "Everest", lat: 27.9881, lon: 86.925, km: 24 },
  { slug: "crater-lake", name: "Crater Lake", lat: 42.9446, lon: -122.109, km: 20 },
  { slug: "vesuvius", name: "Vesuvius", lat: 40.8214, lon: 14.426, km: 12 },
  { slug: "arthurs-seat", name: "Arthur's Seat", lat: 55.9441, lon: -3.1618, km: 5 },
];

const WIDTH = 1200;
const HEIGHT = 160;
const STEP = 3; // sampling resolution in SVG units
const TARGET_LINES = 36; // roughly how many contours to draw
const INDEX_EVERY = 5; // every nth contour is drawn heavier, like a real map
const TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium";

const THEMES = {
  light: { stroke: "#7a6450", minor: 0.5, index: 0.9 },
  dark: { stroke: "#cbbba8", minor: 0.4, index: 0.75 },
};

type Point = [number, number];
type Grid = { nx: number; ny: number; values: Float64Array };

// --- elevation tiles ---------------------------------------------------------

// Minimal PNG decoder for 8-bit RGB/RGBA, which is all Terrarium tiles use.
function decodePng(buf: Buffer): { width: number; height: number; bpp: number; data: Uint8Array } {
  let width = 0;
  let height = 0;
  let bpp = 0;
  const idat: Buffer[] = [];
  for (let o = 8; o < buf.length; ) {
    const len = buf.readUInt32BE(o);
    const type = buf.toString("ascii", o + 4, o + 8);
    const body = buf.subarray(o + 8, o + 8 + len);
    if (type === "IHDR") {
      width = body.readUInt32BE(0);
      height = body.readUInt32BE(4);
      const colourType = body[9];
      if (body[8] !== 8 || (colourType !== 2 && colourType !== 6)) throw new Error("unsupported PNG");
      bpp = colourType === 6 ? 4 : 3;
    } else if (type === "IDAT") {
      idat.push(body);
    }
    o += 12 + len;
  }

  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * bpp;
  const data = new Uint8Array(height * stride);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const row = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let x = 0; x < stride; x++) {
      const i = y * stride + x;
      const a = x >= bpp ? data[i - bpp] : 0;
      const b = y > 0 ? data[i - stride] : 0;
      const c = x >= bpp && y > 0 ? data[i - stride - bpp] : 0;
      let v = row[x];
      if (filter === 1) v += a;
      else if (filter === 2) v += b;
      else if (filter === 3) v += (a + b) >> 1;
      else if (filter === 4) {
        const p = a + b - c;
        const pa = Math.abs(p - a);
        const pb = Math.abs(p - b);
        const pc = Math.abs(p - c);
        v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      data[i] = v & 255;
    }
  }
  return { width, height, bpp, data };
}

async function fetchTile(z: number, x: number, y: number): Promise<Float32Array> {
  const res = await fetch(`${TILE_URL}/${z}/${x}/${y}.png`);
  if (!res.ok) throw new Error(`tile ${z}/${x}/${y}: ${res.status}`);
  const png = decodePng(Buffer.from(await res.arrayBuffer()));
  const out = new Float32Array(png.width * png.height);
  for (let i = 0; i < out.length; i++) {
    const [r, g, b] = png.data.subarray(i * png.bpp, i * png.bpp + 3);
    out[i] = r * 256 + g + b / 256 - 32768;
  }
  return out;
}

// Samples elevation on a WIDTH × HEIGHT grid centred on the place, in Web
// Mercator, choosing the tile zoom that just exceeds the sample spacing.
async function sampleGrid(place: Place): Promise<Grid> {
  const nx = Math.ceil(WIDTH / STEP) + 1;
  const ny = Math.ceil(HEIGHT / STEP) + 1;
  const metresPerSample = (place.km * 1000) / (nx - 1);
  const metresPerPixel = (z: number) => (156543.03392 * Math.cos((place.lat * Math.PI) / 180)) / 2 ** z;
  let z = 0;
  while (z < 15 && metresPerPixel(z) > metresPerSample) z++;

  const scale = 256 * 2 ** z;
  const phi = (place.lat * Math.PI) / 180;
  const cx = ((place.lon + 180) / 360) * scale;
  const cy = ((1 - Math.log(Math.tan(phi) + 1 / Math.cos(phi)) / Math.PI) / 2) * scale;
  const spacing = metresPerSample / metresPerPixel(z);
  const x0 = cx - ((nx - 1) / 2) * spacing;
  const y0 = cy - ((ny - 1) / 2) * spacing;

  const tiles = new Map<string, Float32Array>();
  const loads: Promise<void>[] = [];
  for (let ty = Math.floor(y0 / 256); ty <= Math.floor((y0 + ny * spacing) / 256); ty++) {
    for (let tx = Math.floor(x0 / 256); tx <= Math.floor((x0 + nx * spacing) / 256); tx++) {
      loads.push(fetchTile(z, tx, ty).then((t) => void tiles.set(`${tx}/${ty}`, t)));
    }
  }
  await Promise.all(loads);

  const pixel = (px: number, py: number) => {
    const tile = tiles.get(`${Math.floor(px / 256)}/${Math.floor(py / 256)}`)!;
    return tile[(py & 255) * 256 + (px & 255)];
  };
  const bilinear = (x: number, y: number) => {
    const ix = Math.floor(x);
    const iy = Math.floor(y);
    const fx = x - ix;
    const fy = y - iy;
    const top = pixel(ix, iy) * (1 - fx) + pixel(ix + 1, iy) * fx;
    const bottom = pixel(ix, iy + 1) * (1 - fx) + pixel(ix + 1, iy + 1) * fx;
    return top * (1 - fy) + bottom * fy;
  };

  let values = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      // Sea floor is flattened so coastlines read as the lowest contour.
      values[j * nx + i] = Math.max(0, bilinear(x0 + i * spacing, y0 + j * spacing));
    }
  }

  // A light blur takes the stair-stepping out of upsampled elevation data.
  for (let pass = 0; pass < 2; pass++) {
    const blurred = new Float64Array(values.length);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        let sum = 0;
        let n = 0;
        for (let dj = -1; dj <= 1; dj++) {
          for (let di = -1; di <= 1; di++) {
            const ii = i + di;
            const jj = j + dj;
            if (ii < 0 || jj < 0 || ii >= nx || jj >= ny) continue;
            sum += values[jj * nx + ii];
            n++;
          }
        }
        blurred[j * nx + i] = sum / n;
      }
    }
    values = blurred;
  }
  return { nx, ny, values };
}

// Picks a round contour interval that gives roughly TARGET_LINES lines.
function contourInterval(grid: Grid): { start: number; interval: number; end: number } {
  let min = Infinity;
  let max = -Infinity;
  for (const v of grid.values) {
    min = Math.min(min, v);
    max = Math.max(max, v);
  }
  const ideal = (max - min) / TARGET_LINES;
  const nice = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000];
  const interval = nice.find((n) => n >= ideal) ?? 1000;
  return { start: Math.floor(min / interval + 1) * interval, interval, end: max };
}

// --- marching squares --------------------------------------------------------

function contour(grid: Grid, level: number): Point[][] {
  const { nx, ny, values } = grid;
  const at = (i: number, j: number) => values[j * nx + i];

  // Each crossing point lives on a unique grid edge, so edge ids let us stitch
  // segments into continuous lines without floating point comparisons.
  const hEdge = (i: number, j: number) => (j * nx + i) * 2;
  const vEdge = (i: number, j: number) => (j * nx + i) * 2 + 1;
  const points = new Map<number, Point>();
  const crossing = (edge: number): number => {
    if (!points.has(edge)) {
      const k = edge >> 1;
      const i = k % nx;
      const j = Math.floor(k / nx);
      const [i2, j2] = edge & 1 ? [i, j + 1] : [i + 1, j];
      const a = at(i, j);
      const t = (level - a) / (at(i2, j2) - a);
      points.set(edge, [(i + (i2 - i) * t) * STEP, (j + (j2 - j) * t) * STEP]);
    }
    return edge;
  };

  const segments: [number, number][] = [];
  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const tl = at(i, j);
      const tr = at(i + 1, j);
      const br = at(i + 1, j + 1);
      const bl = at(i, j + 1);
      const c = (tl > level ? 8 : 0) | (tr > level ? 4 : 0) | (br > level ? 2 : 0) | (bl > level ? 1 : 0);
      if (c === 0 || c === 15) continue;

      const top = () => crossing(hEdge(i, j));
      const right = () => crossing(vEdge(i + 1, j));
      const bottom = () => crossing(hEdge(i, j + 1));
      const left = () => crossing(vEdge(i, j));
      const centreHigh = (tl + tr + br + bl) / 4 > level;

      switch (c) {
        case 1: case 14: segments.push([left(), bottom()]); break;
        case 2: case 13: segments.push([bottom(), right()]); break;
        case 3: case 12: segments.push([left(), right()]); break;
        case 4: case 11: segments.push([top(), right()]); break;
        case 6: case 9: segments.push([top(), bottom()]); break;
        case 7: case 8: segments.push([left(), top()]); break;
        case 5:
          if (centreHigh) segments.push([left(), top()], [bottom(), right()]);
          else segments.push([left(), bottom()], [top(), right()]);
          break;
        case 10:
          if (centreHigh) segments.push([left(), bottom()], [top(), right()]);
          else segments.push([left(), top()], [bottom(), right()]);
          break;
      }
    }
  }

  // Stitch segments into polylines.
  const byEdge = new Map<number, number[]>();
  segments.forEach(([a, b], s) => {
    for (const e of [a, b]) {
      const list = byEdge.get(e);
      if (list) list.push(s);
      else byEdge.set(e, [s]);
    }
  });
  const used = new Uint8Array(segments.length);
  const extend = (chain: number[]) => {
    for (;;) {
      const end = chain[chain.length - 1];
      const next = byEdge.get(end)?.find((s) => !used[s]);
      if (next === undefined) return;
      used[next] = 1;
      const [a, b] = segments[next];
      chain.push(a === end ? b : a);
    }
  };

  const lines: Point[][] = [];
  segments.forEach(([a, b], s) => {
    if (used[s]) return;
    used[s] = 1;
    const forward = [a, b];
    extend(forward);
    const backward = [a];
    extend(backward);
    const chain = [...backward.reverse(), ...forward.slice(1)];
    lines.push(chain.map((e) => points.get(e)!));
  });
  return lines;
}

// --- path output -------------------------------------------------------------

function simplify(pts: Point[], tolerance: number): Point[] {
  if (pts.length < 3) return pts;
  const [ax, ay] = pts[0];
  const [bx, by] = pts[pts.length - 1];
  const dx = bx - ax;
  const dy = by - ay;
  const len = Math.hypot(dx, dy);
  let maxDist = 0;
  let index = 0;
  for (let i = 1; i < pts.length - 1; i++) {
    const [px, py] = pts[i];
    const d = len === 0 ? Math.hypot(px - ax, py - ay) : Math.abs(dy * px - dx * py + bx * ay - by * ax) / len;
    if (d > maxDist) {
      maxDist = d;
      index = i;
    }
  }
  if (maxDist <= tolerance) return [pts[0], pts[pts.length - 1]];
  return [...simplify(pts.slice(0, index + 1), tolerance).slice(0, -1), ...simplify(pts.slice(index), tolerance)];
}

const fmt = (n: number) => String(Math.round(n * 10) / 10);
const pt = ([x, y]: Point) => `${fmt(x)} ${fmt(y)}`;
const mid = ([ax, ay]: Point, [bx, by]: Point): Point => [(ax + bx) / 2, (ay + by) / 2];

// Quadratic curves through segment midpoints give smooth lines from a polyline.
function toPath(line: Point[]): string {
  const pts = simplify(line, 0.35);
  if (pts.length < 3) return `M${pts.map(pt).join("L")}`;
  let d = `M${pt(pts[0])}L${pt(mid(pts[0], pts[1]))}`;
  for (let i = 1; i < pts.length - 1; i++) d += `Q${pt(pts[i])} ${pt(mid(pts[i], pts[i + 1]))}`;
  return d + `L${pt(pts[pts.length - 1])}`;
}

const escape = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/'/g, "&#39;");

function render(place: Place, minor: string[], index: string[], theme: (typeof THEMES)[keyof typeof THEMES]): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${WIDTH} ${HEIGHT}" width="${WIDTH}" height="${HEIGHT}" fill="none" stroke="${theme.stroke}" stroke-linecap="round" stroke-linejoin="round">
<title>${escape(`Contour map of ${place.name}`)}</title>
<defs>
<linearGradient id="fx"><stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset=".12" stop-color="#fff"/><stop offset=".88" stop-color="#fff"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
<mask id="m"><rect width="${WIDTH}" height="${HEIGHT}" fill="url(#fx)"/></mask>
</defs>
<g mask="url(#m)">
<path stroke-width="1" stroke-opacity="${theme.minor}" d="${minor.join("")}"/>
<path stroke-width="1.6" stroke-opacity="${theme.index}" d="${index.join("")}"/>
</g>
</svg>
`;
}

// --- main --------------------------------------------------------------------

const placeArg = process.argv.indexOf("--place");
const day = Math.floor(Date.now() / 86_400_000);
const place =
  placeArg === -1 ? PLACES[day % PLACES.length] : PLACES.find((p) => p.slug === process.argv[placeArg + 1]);
if (!place) throw new Error(`unknown place; try one of: ${PLACES.map((p) => p.slug).join(", ")}`);

const grid = await sampleGrid(place);
const { start, interval, end } = contourInterval(grid);

const minor: string[] = [];
const index: string[] = [];
for (let level = start; level < end; level += interval) {
  const paths = contour(grid, level).map(toPath);
  (level % (interval * INDEX_EVERY) === 0 ? index : minor).push(...paths);
}

mkdirSync("assets", { recursive: true });
for (const [name, theme] of Object.entries(THEMES)) {
  writeFileSync(`assets/contours-${name}.svg`, render(place, minor, index, theme));
}
console.log(`${place.name}: ${interval} m contours → assets/contours-{light,dark}.svg`);
