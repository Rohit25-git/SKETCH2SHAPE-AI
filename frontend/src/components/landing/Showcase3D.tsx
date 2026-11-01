import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Float } from '@react-three/drei';
import * as THREE from 'three';

const OrganicShape: React.FC = () => {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.getElapsedTime() * 0.4;
      meshRef.current.rotation.x = Math.sin(state.clock.getElapsedTime() * 0.2) * 0.2;
    }
  });

  return (
    <Float speed={2} rotationIntensity={0.5} floatIntensity={1}>
      <mesh ref={meshRef}>
        <torusKnotGeometry args={[1.2, 0.38, 128, 32]} />
        <meshPhysicalMaterial
          color="#8B5CF6"
          roughness={0.15}
          metalness={0.8}
          clearcoat={1.0}
          clearcoatRoughness={0.1}
          emissive="#2E1065"
          emissiveIntensity={0.4}
        />
      </mesh>
    </Float>
  );
};

export const Showcase3D: React.FC = () => {
  return (
    <div className="relative w-full h-[400px] md:h-[480px] rounded-3xl bg-gradient-to-b from-[#F8FAFC] to-[#EEF2F6] border border-slate-200 overflow-hidden shadow-studio-card backdrop-blur-xl">
      <Canvas
        camera={{ position: [0, 0, 4.2], fov: 45 }}
        gl={{ antialias: true, alpha: true }}
      >
        <ambientLight intensity={0.9} />
        <directionalLight position={[4, 6, 4]} intensity={1.5} color="#FFFFFF" />
        <pointLight position={[-4, -2, -2]} intensity={0.8} color="#0284C7" />
        <pointLight position={[2, -4, 2]} intensity={0.6} color="#6366F1" />
        
        <OrganicShape />

        <OrbitControls
          enableZoom={false}
          autoRotate
          autoRotateSpeed={1.2}
          enablePan={false}
        />
      </Canvas>

      <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between pointer-events-none px-4 py-2 rounded-2xl bg-white/90 backdrop-blur border border-slate-200 shadow-xs">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-cyan-accent animate-ping" />
          <span className="text-xs font-bold text-slate-900">Interactive 3D Viewport</span>
        </div>
        <span className="text-[11px] text-slate-500 font-medium">Drag to Rotate</span>
      </div>
    </div>
  );
};
