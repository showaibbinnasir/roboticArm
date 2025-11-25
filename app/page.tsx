"use client"

import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';

// ---- Constants & types ----
const LINK_LENGTH = 49.5;
const BASE_HEIGHT = 25;
const FLOOR_Z = 0;
const CAMERA_DIST = 280;
const DEFAULT_CAM_THETA = 270;
const DEFAULT_CAM_PHI = 20;

type JointKey = 'q1' | 'q2' | 'q3' | 'q4' | 'gripper';

type JointLimits = Record<JointKey, { min: number; max: number; label: string; default: number }>;

const JOINT_LIMITS: JointLimits = {
  q1: { min: 0, max: 180, label: 'Yaw (Base)', default: 90 },
  q2: { min: 90, max: 180, label: 'Link 1', default: 90 },
  q3: { min: 0, max: 180, label: 'Link 2', default: 135 },
  q4: { min: 0, max: 180, label: 'Link 3', default: 135 },
  gripper: { min: 80, max: 120, label: 'Gripper', default: 80 }
};

const deg2rad = (d: number) => d * Math.PI / 180;
const rad2deg = (r: number) => r * 180 / Math.PI;

// FK/IK utilities (kept strongly-typed)
const calculateFK = (q1: number, q2: number, q3: number, q4: number) => {
  const t1 = deg2rad(q1), t2 = deg2rad(q2 - 90), t3 = deg2rad(q3 - 90), t4 = deg2rad(q4 - 90);
  const p0 = { x: 0, y: 0, z: 0 };
  const p1 = { x: 0, y: 0, z: BASE_HEIGHT };
  const r1 = LINK_LENGTH * Math.cos(t2);
  const z1 = BASE_HEIGHT + LINK_LENGTH * Math.sin(t2);
  const p2 = { x: Math.cos(t1) * r1, y: Math.sin(t1) * r1, z: z1 };
  const cumAngle2 = t2 + t3;
  const r2 = r1 + LINK_LENGTH * Math.cos(cumAngle2);
  const z2 = z1 + LINK_LENGTH * Math.sin(cumAngle2);
  const p3 = { x: Math.cos(t1) * r2, y: Math.sin(t1) * r2, z: z2 };
  const cumAngle3 = cumAngle2 + t4;
  const r3 = r2 + LINK_LENGTH * Math.cos(cumAngle3);
  const z3 = z2 + LINK_LENGTH * Math.sin(cumAngle3);
  const p4 = { x: Math.cos(t1) * r3, y: Math.sin(t1) * r3, z: z3 };
  const eeDir = { x: Math.cos(t1) * Math.cos(cumAngle3), y: Math.sin(t1) * Math.cos(cumAngle3), z: Math.sin(cumAngle3) };
  return { points: [p0, p1, p2, p3, p4], eeDir, yaw: t1 };
};

const solveIK = (tx: number, ty: number, tz: number, curr: Record<JointKey, number>) => {
  let q = { ...curr };

  const rhoT = Math.sqrt(tx * tx + ty * ty);
  if (rhoT > 0.1) {
    q.q1 = rad2deg(Math.atan2(ty, tx));
    if (q.q1 < 0) q.q1 += 360;
    if (q.q1 > 180) q.q1 = 180 - (q.q1 - 180);
  }
  q.q1 = Math.max(0, Math.min(180, q.q1));

  const zT = tz;

  const planarFK = (q2: number, q3: number, q4: number) => {
    const t2 = deg2rad(q2 - 90), t3 = deg2rad(q3 - 90), t4 = deg2rad(q4 - 90);
    return {
      rho: LINK_LENGTH * (Math.cos(t2) + Math.cos(t2 + t3) + Math.cos(t2 + t3 + t4)),
      z: BASE_HEIGHT + LINK_LENGTH * (Math.sin(t2) + Math.sin(t2 + t3) + Math.sin(t2 + t3 + t4))
    };
  };

  const planarJacobian = (q2: number, q3: number, q4: number) => {
    const t2 = deg2rad(q2 - 90), t3 = deg2rad(q3 - 90), t4 = deg2rad(q4 - 90);
    const s2 = Math.sin(t2), c2 = Math.cos(t2);
    const s23 = Math.sin(t2 + t3), c23 = Math.cos(t2 + t3);
    const s234 = Math.sin(t2 + t3 + t4), c234 = Math.cos(t2 + t3 + t4);

    const dr2 = -LINK_LENGTH * (s2 + s23 + s234);
    const dr3 = -LINK_LENGTH * (s23 + s234);
    const dr4 = -LINK_LENGTH * s234;

    const dz2 = LINK_LENGTH * (c2 + c23 + c234);
    const dz3 = LINK_LENGTH * (c23 + c234);
    const dz4 = LINK_LENGTH * c234;

    return { dr: [dr2, dr3, dr4], dz: [dz2, dz3, dz4] };
  };

  const damping = 0.5;
  const stepScale = 0.8;

  for (let iter = 0; iter < 300; iter++) {
    const c = planarFK(q.q2, q.q3, q.q4);
    const errRho = rhoT - c.rho;
    const errZ = zT - c.z;
    const errNorm = Math.sqrt(errRho * errRho + errZ * errZ);

    if (errNorm < 0.5) {
      return { joints: q, success: true, error: errNorm };
    }

    const J = planarJacobian(q.q2, q.q3, q.q4);

    const JJT00 = J.dr[0] * J.dr[0] + J.dr[1] * J.dr[1] + J.dr[2] * J.dr[2] + damping * damping;
    const JJT01 = J.dr[0] * J.dz[0] + J.dr[1] * J.dz[1] + J.dr[2] * J.dz[2];
    const JJT11 = J.dz[0] * J.dz[0] + J.dz[1] * J.dz[1] + J.dz[2] * J.dz[2] + damping * damping;

    const det = JJT00 * JJT11 - JJT01 * JJT01;
    if (Math.abs(det) < 1e-10) break;

    const inv00 = JJT11 / det, inv01 = -JJT01 / det, inv11 = JJT00 / det;

    const tmp0 = inv00 * errRho + inv01 * errZ;
    const tmp1 = inv01 * errRho + inv11 * errZ;

    const dq2 = rad2deg((J.dr[0] * tmp0 + J.dz[0] * tmp1)) * stepScale;
    const dq3 = rad2deg((J.dr[1] * tmp0 + J.dz[1] * tmp1)) * stepScale;
    const dq4 = rad2deg((J.dr[2] * tmp0 + J.dz[2] * tmp1)) * stepScale;

    q.q2 = Math.max(90, Math.min(180, q.q2 + dq2));
    q.q3 = Math.max(0, Math.min(180, q.q3 + dq3));
    q.q4 = Math.max(0, Math.min(180, q.q4 + dq4));
  }

  const finalPos = planarFK(q.q2, q.q3, q.q4);
  const finalErr = Math.sqrt((rhoT - finalPos.rho) ** 2 + (zT - finalPos.z) ** 2);
  return { joints: q, success: finalErr < 5, error: finalErr };
};

const checkCollision = (pts: Array<{ x: number; y: number; z: number }>, eeDir: { x: number; y: number; z: number }) => {
  const cols: number[] = [];
  for (let i = 1; i < pts.length; i++) if (pts[i].z < -0.1) cols.push(i);
  if (pts[4].z + eeDir.z * 25 < -0.1) cols.push(5);
  return cols;
};

const checkReach = (x: number, y: number, z: number) => {
  const d = Math.sqrt(x * x + y * y + (z - BASE_HEIGHT) * (z - BASE_HEIGHT));
  if (d > LINK_LENGTH * 3) return 'Target too far';
  if (z < 0) return 'Target below floor';
  return null;
};

// ---- React component ----
export default function Home(): JSX.Element {
  // Refs with proper types to avoid runtime errors in SSR/production
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const robotRef = useRef<{
    js: THREE.Mesh[];
    ls: THREE.Mesh[];
    g: { base: THREE.Mesh; f1: THREE.Mesh; f2: THREE.Mesh };
  } | null>(null);
  const targetRef = useRef<THREE.Mesh | null>(null);
  const frameRef = useRef<number | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);

  const [joints, setJoints] = useState<Record<JointKey, number>>({ q1: 90, q2: 90, q3: 135, q4: 135, gripper: 80 });
  const [collision, setCollision] = useState<boolean>(false);
  const [eePos, setEePos] = useState<{ x: number; y: number; z: number }>({ x: 0, y: 0, z: 0 });
  const [camTheta, setCamTheta] = useState<number>(DEFAULT_CAM_THETA);
  const [camPhi, setCamPhi] = useState<number>(DEFAULT_CAM_PHI);
  const [mode, setMode] = useState<'fk' | 'ik'>('fk');
  const [target, setTarget] = useState<{ x: number; y: number; z: number }>({ x: 50, y: 0, z: 100 });
  const [ikMsg, setIkMsg] = useState<string>('');
  const [esp32Ip, setEsp32Ip] = useState<string>('192.168.1.100');
  const [status, setStatus] = useState<string>('disconnected');
  const [lastCmd, setLastCmd] = useState<Record<string, number> | null>(null);
  const [encrypt, setEncrypt] = useState<boolean>(true);

  const updateCam = useCallback((th: number, ph: number) => {
    if (!cameraRef.current) return;
    const t = deg2rad(th), p = deg2rad(ph);
    cameraRef.current.position.set(CAMERA_DIST * Math.cos(p) * Math.cos(t), CAMERA_DIST * Math.cos(p) * Math.sin(t), CAMERA_DIST * Math.sin(p) + 60);
    cameraRef.current.lookAt(0, 0, 50);
    cameraRef.current.up.set(0, 0, 1);
  }, []);

  useEffect(() => {
    if (!containerRef.current) return;
    // size relies on client DOM
    const w = containerRef.current.clientWidth, h = 400;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
    const camera = new THREE.PerspectiveCamera(50, w / h, 1, 1000);
    cameraRef.current = camera;
    updateCam(DEFAULT_CAM_THETA, DEFAULT_CAM_PHI);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(w, h);
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    scene.add(new THREE.AmbientLight(0xffffff, 0.4));
    const dl = new THREE.DirectionalLight(0xffffff, 0.8);
    dl.position.set(100, 100, 200);
    scene.add(dl);

    const floor = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), new THREE.MeshStandardMaterial({ color: 0x2d3436 }));
    floor.rotation.x = -Math.PI / 2;
    floor.rotation.z = Math.PI / 2;
    scene.add(floor);
    const grid = new THREE.GridHelper(400, 20, 0x444444, 0x333333);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);

    const base = new THREE.Mesh(new THREE.CylinderGeometry(15, 18, BASE_HEIGHT, 16), new THREE.MeshStandardMaterial({ color: 0x555555 }));
    base.position.set(0, 0, BASE_HEIGHT / 2);
    base.rotation.x = Math.PI / 2;
    scene.add(base);

    const jGeo = new THREE.SphereGeometry(8, 16, 16), jMat = new THREE.MeshStandardMaterial({ color: 0xcccccc });
    const js = [new THREE.Mesh(jGeo, jMat), new THREE.Mesh(jGeo, jMat), new THREE.Mesh(jGeo, jMat)];
    js.forEach(j => scene.add(j));

    const lGeo = new THREE.CylinderGeometry(5, 5, LINK_LENGTH, 8);
    const ls = [
      new THREE.Mesh(lGeo, new THREE.MeshStandardMaterial({ color: 0x3498db })),
      new THREE.Mesh(lGeo, new THREE.MeshStandardMaterial({ color: 0x2ecc71 })),
      new THREE.Mesh(lGeo, new THREE.MeshStandardMaterial({ color: 0xf39c12 }))
    ];
    ls.forEach(l => scene.add(l));

    const gMat = new THREE.MeshStandardMaterial({ color: 0xe91e63 });
    const gBase = new THREE.Mesh(new THREE.BoxGeometry(12, 12, 8), gMat);
    const f1 = new THREE.Mesh(new THREE.BoxGeometry(4, 3, 20), gMat);
    const f2 = new THREE.Mesh(new THREE.BoxGeometry(4, 3, 20), gMat);
    scene.add(gBase, f1, f2);

    const tgt = new THREE.Mesh(new THREE.SphereGeometry(6, 16, 16), new THREE.MeshStandardMaterial({ color: 0xff0000, transparent: true, opacity: 0.7 }));
    tgt.visible = false;
    scene.add(tgt);

    robotRef.current = { js, ls, g: { base: gBase, f1, f2 } };
    targetRef.current = tgt;

    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
      if (containerRef.current && renderer.domElement.parentElement === containerRef.current) {
        containerRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
      // small defensive cleanup
      robotRef.current = null;
      cameraRef.current = null;
      rendererRef.current = null;
    };
  }, [updateCam]);

  useEffect(() => { updateCam(camTheta, camPhi); }, [camTheta, camPhi, updateCam]);

  // Update THREE objects when joints change
  useEffect(() => {
    const { points: pts, eeDir, yaw } = calculateFK(joints.q1, joints.q2, joints.q3, joints.q4);
    const r = robotRef.current;
    if (!r || !r.ls) return;
    const cols = checkCollision(pts, eeDir);
    setCollision(cols.length > 0);
    setEePos(pts[4]);

    r.js[0].position.set(pts[1].x, pts[1].y, pts[1].z);
    r.js[1].position.set(pts[2].x, pts[2].y, pts[2].z);
    r.js[2].position.set(pts[3].x, pts[3].y, pts[3].z);

    for (let i = 0; i < 3; i++) {
      const p1 = pts[i + 1], p2 = pts[i + 2];
      r.ls[i].position.set((p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2);
      const dir = new THREE.Vector3(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z).normalize();
      r.ls[i].setRotationFromQuaternion(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir));
      // safe set color
      try {
        (r.ls[i].material as THREE.MeshStandardMaterial).color.setHex(cols.includes(i + 2) ? 0xff0000 : [0x3498db, 0x2ecc71, 0xf39c12][i]);
      } catch (e) {
        // ignore coloring errors in some WebGL contexts
      }
    }

    const ee = pts[4], gCol = cols.includes(5), gC = gCol ? 0xff0000 : 0xe91e63;
    try {
      (r.g.base.material as THREE.MeshStandardMaterial).color.setHex(gC);
      (r.g.f1.material as THREE.MeshStandardMaterial).color.setHex(gC);
      (r.g.f2.material as THREE.MeshStandardMaterial).color.setHex(gC);
    } catch (e) {
      // ignore
    }
    r.g.base.position.set(ee.x, ee.y, ee.z);
    const gDir = new THREE.Vector3(eeDir.x, eeDir.y, eeDir.z).normalize();
    const gQ = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), gDir);
    r.g.base.setRotationFromQuaternion(gQ);
    const sp = (joints.gripper - 80) / 40 * 10 + 5, fo = 14, px = -Math.sin(yaw), py = Math.cos(yaw);
    r.g.f1.position.set(ee.x + eeDir.x * fo + px * sp, ee.y + eeDir.y * fo + py * sp, ee.z + eeDir.z * fo);
    r.g.f2.position.set(ee.x + eeDir.x * fo - px * sp, ee.y + eeDir.y * fo - py * sp, ee.z + eeDir.z * fo);
    r.g.f1.setRotationFromQuaternion(gQ);
    r.g.f2.setRotationFromQuaternion(gQ);
  }, [joints]);

  useEffect(() => {
    if (targetRef.current) {
      targetRef.current.visible = mode === 'ik';
      targetRef.current.position.set(target.x, target.y, target.z);
    }
  }, [mode, target]);

  // Helpers to set values safely
  const setJ = (k: JointKey, v: number | string) => {
    const n = typeof v === 'string' ? Number(v) : v;
    setJoints(p => ({ ...p, [k]: Math.max(JOINT_LIMITS[k].min, Math.min(JOINT_LIMITS[k].max, Number.isFinite(n) ? n : p[k])) }));
  };
  const setT = (k: keyof typeof target, v: number | string) => {
    const n = typeof v === 'string' ? Number(v) : v;
    setTarget(p => ({ ...p, [k]: Number.isFinite(n) ? n : p[k] }));
  };

  const calcIK = (): boolean => {
    const err = checkReach(target.x, target.y, target.z);
    if (err) { setIkMsg('❌ ' + err); return false; }

    const res = solveIK(target.x, target.y, target.z, joints as Record<JointKey, number>);

    if (res.success) {
      setJoints(p => ({ ...p, q1: res.joints.q1, q2: res.joints.q2, q3: res.joints.q3, q4: res.joints.q4 }));
      setIkMsg(`✅ IK solved! Error: ${res.error.toFixed(2)}mm`);
      return true;
    } else {
      setJoints(p => ({ ...p, q1: res.joints.q1, q2: res.joints.q2, q3: res.joints.q3, q4: res.joints.q4 }));
      setIkMsg(`⚠️ IK partial (error: ${res.error.toFixed(1)}mm) - may not reach exactly`);
      return res.error < 10;
    }
  };

  const calcIKAndMove = async () => {
    const success = calcIK();
    if (success) {
      // small delay to allow React to flush state (keeps UX similar to original)
      setTimeout(() => send(), 100);
    }
  };

  const send = () => {
    if (collision) { alert('Collision! Cannot send.'); return; }
    const cmd = { q1: Math.round(joints.q1), q2: Math.round(joints.q2), q3: Math.round(joints.q3), q4: Math.round(joints.q4), gripper: Math.round(joints.gripper) };
    // Keep fetch but handle success/error and persist lastCmd/status for UI
    fetch('https://final-server-mocha.vercel.app/postValue', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cmd })
    }).then(async res => {
      if (res.ok) {
        setStatus('sent');
        setLastCmd(cmd);
      } else {
        setStatus('error');
        console.error('Failed to send command to server', await res.text());
      }
    }).catch(err => {
      setStatus('error');
      console.error(err);
    });
  };

  return (
    <div className="bg-gray-900 text-white p-2 text-xs">
      <h1 className="text-sm font-bold text-center mb-1">5-DOF Robot Arm Controller</h1>
      {collision && <div className="bg-red-600 p-1 rounded mb-1 text-center font-bold">⚠️ COLLISION!</div>}
      <div className='flex justify-center mb-2'>
        <div ref={containerRef} className="w-[550px]  border b-6 border-white rounded mb-2 flex justify-center" />
      </div>

      <div className="flex gap-1 mb-2">
        <button onClick={() => setMode('fk')} className={`flex-1 py-1 rounded font-bold ${mode === 'fk' ? 'bg-blue-600' : 'bg-gray-700'}`}>FK Mode</button>
        <button onClick={() => setMode('ik')} className={`flex-1 py-1 rounded font-bold ${mode === 'ik' ? 'bg-purple-600' : 'bg-gray-700'}`}>IK Mode</button>
      </div>

      {mode === 'fk' && (
        <div className='flex justify-center'>
          <div className="bg-gray-800 w-[75%] p-2 rounded mb-2">
            {Object.entries(JOINT_LIMITS).map(([k, l]) => {
              const key = k as JointKey;
              return (
                <div key={k} className="mb-1">
                  <div className="flex justify-between"><span>{l.label}</span><span>{joints[key].toFixed(0)}°</span></div>
                  <input
                    type="range"
                    min={l.min}
                    max={l.max}
                    value={joints[key]}
                    onChange={e => setJ(key, Number(e.target.value))}
                    className="w-full"
                  />
                </div>
              );
            })}
          </div>
        </div>
      )}

      {mode === 'ik' && (
        <div className="bg-gray-800 p-2 rounded mb-2">
          <div className="grid grid-cols-3 gap-1 mb-2">
            {['x', 'y', 'z'].map(k => (
              <div key={k}>
                <label className="text-gray-400">{k.toUpperCase()} (mm)</label>
                <input type="number" value={(target as any)[k]} onChange={e => setT(k as keyof typeof target, e.target.value)} className="w-full bg-gray-700 px-1 py-1 rounded" />
              </div>
            ))}
          </div>
          <button onClick={calcIK} className="w-full bg-purple-600 py-1 rounded font-bold mb-1">Calculate IK</button>
          <button onClick={calcIKAndMove} className="w-full bg-green-600 py-1 rounded font-bold mb-1">Calculate IK & MOVE</button>
          {ikMsg && <div className={`p-1 rounded text-center ${ikMsg.includes('solved') ? 'bg-green-800' : 'bg-red-800'}`}>{ikMsg}</div>}
          <div className="mt-2">
            <div className="flex justify-between"><span>Gripper</span><span>{joints.gripper}°</span></div>
            <input type="range" min={80} max={120} value={joints.gripper} onChange={e => setJ('gripper', Number(e.target.value))} className="w-full" />
          </div>
        </div>
      )}

      <div className="bg-gray-800 p-2 rounded mb-2">
        <div className="font-bold text-gray-400 mb-1">Camera</div>
        <div className="grid grid-cols-2 gap-2">
          <div><span>Rotation: {camTheta}°</span><input type="range" min={0} max={360} value={camTheta} onChange={e => setCamTheta(+e.target.value)} className="w-full" /></div>
          <div><span>Elevation: {camPhi}°</span><input type="range" min={0} max={89} value={camPhi} onChange={e => setCamPhi(+e.target.value)} className="w-full" /></div>
        </div>
      </div>

      <div className="bg-gray-800 p-2 rounded mb-2">
        <div className="font-bold text-gray-400">End Effector</div>
        <div className="flex justify-around">
          <span>X: {eePos.x.toFixed(1)}</span>
          <span>Y: {eePos.y.toFixed(1)}</span>
          <span className={eePos.z < 0 ? 'text-red-400' : ''}>Z: {eePos.z.toFixed(1)}</span>
        </div>
      </div>

      {lastCmd && <div className="bg-gray-800 p-1 rounded mb-2 text-green-400"><code>{JSON.stringify(lastCmd)}</code></div>}

      <div className="grid grid-cols-2 gap-2">
        <button onClick={send} disabled={collision} className={`py-2 rounded font-bold ${collision ? 'bg-gray-600' : 'bg-green-600'}`}>📡 MOVE</button>
        <button onClick={() => setJoints({ q1: 90, q2: 90, q3: 135, q4: 135, gripper: 80 })} className="py-2 rounded font-bold bg-blue-600">Reset</button>
      </div>
    </div>
  );
}
