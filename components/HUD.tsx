import React from 'react';
import { ResponsiveContainer, AreaChart, Area, YAxis } from 'recharts';
import { ConnectionStatus, QualityProfile, Telemetry } from '../types';

interface HUDProps {
  status: ConnectionStatus;
  telemetry: Telemetry;
  quality: QualityProfile;
  resolution: string;
  latencyHistory: { timestamp: number; value: number }[];
}

export const HUD: React.FC<HUDProps> = ({
  status,
  telemetry,
  quality,
  resolution,
  latencyHistory
}) => {

  const getStatusColor = (s: ConnectionStatus) => {
    switch (s) {
      case ConnectionStatus.CONNECTED: return 'bg-emerald-500';
      case ConnectionStatus.CONNECTING: return 'bg-amber-500';
      case ConnectionStatus.ERROR: return 'bg-red-500';
      default: return 'bg-zinc-500';
    }
  };

  return (
    <div className="absolute inset-0 pointer-events-none p-6 flex flex-col justify-between z-20 font-sans">

      {/* TOP BAR */}
      <div className="flex justify-between items-start">
        <div className="flex gap-4 items-center bg-black/50 backdrop-blur-sm px-3 py-2 rounded-md border border-zinc-800/50">
          <div className={`h-1.5 w-1.5 rounded-full ${getStatusColor(status)}`} />
          <div className="text-[10px] font-medium tracking-widest text-zinc-300 uppercase">
            {status}
          </div>
          <div className="w-px h-3 bg-zinc-800"></div>
          <div className="text-[10px] font-medium tracking-widest text-zinc-400">
            {resolution}
          </div>
          <div className="w-px h-3 bg-zinc-800"></div>
          <div className="text-[10px] font-medium tracking-widest text-zinc-400">
            {quality.toUpperCase()}
          </div>
        </div>

        {/* Latency Graph */}
        <div className="w-40 h-12 min-w-[10rem] min-h-[3rem] bg-black/50 backdrop-blur-sm border border-zinc-800/50 rounded-md overflow-hidden relative">
          <div className="absolute top-1 right-2 text-[9px] font-medium text-zinc-500 z-10 tracking-wider">
            LATENCY
          </div>
          {latencyHistory.length > 1 ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={latencyHistory}>
                <YAxis hide domain={[0, 'auto']} />
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke="#52525b"
                  fill="#27272a"
                  fillOpacity={0.5}
                  strokeWidth={1}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : null}
        </div>
      </div>

      {/* CENTER CROSSHAIR - Minimalist */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="w-1 h-1 bg-white/50 rounded-full"></div>
      </div>

      {/* BOTTOM BAR */}
      <div className="flex justify-between items-end">
        <div className="bg-black/50 backdrop-blur-sm p-4 rounded-md border border-zinc-800/50 space-y-3 min-w-[200px]">
          <div className="text-[9px] font-bold text-zinc-500 uppercase tracking-widest">Performance</div>
          <div className="grid grid-cols-2 gap-x-8 gap-y-2">
            <Stat label="FPS" value={telemetry.fps.toFixed(0)} />
            <Stat label="Buffer" value={`${telemetry.bufferMs.toFixed(0)}ms`} />
            <Stat label="RTT" value={`${telemetry.latencyMs.toFixed(0)}ms`} />
            <Stat label="Gen" value={`${telemetry.generationTimeMs.toFixed(0)}ms`} />
          </div>
        </div>

        <div className="bg-black/50 backdrop-blur-sm p-4 rounded-md border border-zinc-800/50">
          <div className="flex gap-4 text-[10px] font-medium text-zinc-400">
            <div className="flex items-center gap-1.5"><K>WASD</K> <span>Move</span></div>
            <div className="w-px h-3 bg-zinc-800 my-auto"></div>
            <div className="flex items-center gap-1.5"><K>Mouse</K> <span>Look</span></div>
            <div className="w-px h-3 bg-zinc-800 my-auto"></div>
            <div className="flex items-center gap-1.5"><K>Space</K> <span>Toggle Idle</span></div>
          </div>
        </div>
      </div>
    </div>
  );
};

const Stat: React.FC<{ label: string, value: string }> = ({ label, value }) => (
  <div className="flex flex-col">
    <span className="text-[9px] text-zinc-500 uppercase tracking-wider font-medium">{label}</span>
    <span className="text-xs text-zinc-200 font-mono">{value}</span>
  </div>
);

const K: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <span className="px-1.5 py-0.5 bg-zinc-800 border border-zinc-700 rounded text-[9px] text-zinc-300 font-mono">
    {children}
  </span>
);