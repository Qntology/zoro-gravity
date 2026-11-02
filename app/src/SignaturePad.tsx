import { useEffect, useRef } from "react";

/** 서명 측정값. 코어 `Signature` 와 같은 이름입니다. */
export interface SignatureStats {
  stroke_count: number;
  bbox_area_ratio: number;
  single_line: boolean;
}

interface Point {
  x: number;
  y: number;
}

const HEIGHT = 130;
/** 한 줄 직선으로 볼 최대 흔들림 (px) */
const LINE_TOLERANCE = 4;

/** 한 획이 직선에 가까운지 봅니다. 두 점 사이 직선에서 가장 먼 점의 거리로 잽니다. */
function nearlyStraight(points: Point[]): boolean {
  if (points.length < 3) return true;
  const a = points[0];
  const b = points[points.length - 1];
  const len = Math.hypot(b.x - a.x, b.y - a.y) || 1;
  const far = Math.max(...points.map((p) => Math.abs((b.y - a.y) * p.x - (b.x - a.x) * p.y + b.x * a.y - b.y * a.x) / len));
  return far < LINE_TOLERANCE;
}

/** 터치나 마우스로 서명하는 패드. 획수, 면적, 직선 여부를 `onChange` 로 알려 줍니다. 판정은 코어가 합니다. */
export function SignaturePad({ onChange }: { onChange: (stats: SignatureStats | null) => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const strokes = useRef<Point[][]>([]);
  const drawing = useRef(false);

  const report = () => {
    const canvas = canvasRef.current;
    const done = strokes.current.filter((s) => s.length > 1);
    if (!canvas || done.length === 0) {
      onChange(null);
      return;
    }
    const all = done.flat();
    const xs = all.map((p) => p.x);
    const ys = all.map((p) => p.y);
    const area = (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys));
    const total = canvas.clientWidth * HEIGHT;
    onChange({
      stroke_count: done.length,
      bbox_area_ratio: total > 0 ? area / total : 0,
      single_line: done.length === 1 && nearlyStraight(done[0]),
    });
  };

  const redraw = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    ctx.clearRect(0, 0, canvas.clientWidth, HEIGHT);
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.strokeStyle = "#1f2328";
    for (const s of strokes.current) {
      if (s.length === 0) continue;
      ctx.beginPath();
      ctx.moveTo(s[0].x, s[0].y);
      for (const p of s.slice(1)) ctx.lineTo(p.x, p.y);
      ctx.stroke();
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.clientWidth * dpr;
    canvas.height = HEIGHT * dpr;
    canvas.getContext("2d")?.setTransform(dpr, 0, 0, dpr, 0, 0);
  }, []);

  const point = (e: React.PointerEvent<HTMLCanvasElement>): Point => {
    const r = e.currentTarget.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  const clear = () => {
    strokes.current = [];
    redraw();
    onChange(null);
  };

  return (
    <div className="sign">
      <canvas
        ref={canvasRef}
        className="sign-canvas"
        style={{ height: HEIGHT }}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          drawing.current = true;
          strokes.current.push([point(e)]);
        }}
        onPointerMove={(e) => {
          if (!drawing.current) return;
          const s = strokes.current[strokes.current.length - 1];
          s.push(point(e));
          redraw();
        }}
        onPointerUp={() => {
          drawing.current = false;
          report();
        }}
        onPointerCancel={() => {
          drawing.current = false;
          report();
        }}
      />
      <div className="sign-foot">
        <span className="muted">손가락이나 마우스로 성함을 적어 주세요.</span>
        <button type="button" className="plain" onClick={clear}>
          지우고 다시 쓰기
        </button>
      </div>
    </div>
  );
}
