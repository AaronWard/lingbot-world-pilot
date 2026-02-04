import React, { useState } from 'react';
import { QualityProfile, SessionConfig } from '../types';
import { Button } from './Button';
import { DEFAULT_CONFIG } from '../constants';

interface SetupScreenProps {
  onStart: (config: SessionConfig) => void;
}

export const SetupScreen: React.FC<SetupScreenProps> = ({ onStart }) => {
  const [prompt, setPrompt] = useState(DEFAULT_CONFIG.prompt);
  const [quality, setQuality] = useState<QualityProfile>(DEFAULT_CONFIG.quality);
  const [resolution, setResolution] = useState<'480p' | '720p'>('480p');
  const [image, setImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImage(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onStart({
      prompt,
      quality,
      resolution,
      initImage: image
    });
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-black text-foreground relative overflow-hidden">
      {/* Background decoration */}
      <div className="absolute inset-0 pointer-events-none">
         <div className="absolute top-0 left-0 w-full h-96 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-zinc-800/20 via-black to-black"></div>
      </div>

      <div className="relative w-full max-w-[600px] p-6">
        <div className="mb-10 text-center">
          <h1 className="text-4xl font-bold tracking-tight mb-2 text-white">
            LingBot<span className="text-zinc-500">World</span>
          </h1>
          <p className="text-zinc-500 text-sm">
            High-fidelity neural rendering interface.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-8">
          {/* Prompt Section */}
          <div className="space-y-3">
            <label className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Simulation Prompt</label>
            <textarea 
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full h-28 bg-black border border-zinc-800 rounded-lg p-4 text-sm text-zinc-100 placeholder-zinc-700 focus:border-white focus:ring-1 focus:ring-white outline-none transition-colors resize-none font-sans leading-relaxed"
              placeholder="Describe the environment..."
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Image Upload */}
            <div className="space-y-3">
               <label className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Initialization Image</label>
               <div className="relative h-24 border border-zinc-800 rounded-lg border-dashed hover:border-zinc-600 transition-colors bg-black flex flex-col items-center justify-center overflow-hidden group cursor-pointer">
                 <input 
                   type="file" 
                   accept="image/*"
                   onChange={handleImageUpload}
                   className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                 />
                 {previewUrl ? (
                   <img src={previewUrl} alt="Preview" className="absolute inset-0 w-full h-full object-cover opacity-60 group-hover:opacity-100 transition-opacity" />
                 ) : (
                   <div className="text-center p-2">
                     <span className="block text-zinc-500 text-xs font-medium group-hover:text-zinc-300 transition-colors">Upload Source</span>
                   </div>
                 )}
               </div>
            </div>

            {/* Settings */}
            <div className="space-y-6">
               {/* Resolution */}
               <div className="space-y-3">
                <label className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Output Resolution</label>
                <div className="flex bg-zinc-900/50 p-1 rounded-lg border border-zinc-800/50">
                  {(['480p', '720p'] as const).map((res) => (
                    <button
                      key={res}
                      type="button"
                      onClick={() => setResolution(res)}
                      className={`flex-1 py-1 text-xs font-medium rounded-md transition-all ${resolution === res ? 'bg-zinc-800 text-white shadow-sm' : 'text-zinc-500 hover:text-zinc-300'}`}
                    >
                      {res}
                    </button>
                  ))}
                </div>
               </div>

               {/* Quality Profile */}
               <div className="space-y-3">
                <label className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Performance Mode</label>
                <div className="flex bg-zinc-900/50 p-1 rounded-lg border border-zinc-800/50">
                  {Object.values(QualityProfile).map((q) => (
                    <button
                      key={q}
                      type="button"
                      onClick={() => setQuality(q)}
                      className={`flex-1 py-1 text-xs font-medium rounded-md capitalize transition-all ${quality === q ? 'bg-zinc-800 text-white shadow-sm' : 'text-zinc-500 hover:text-zinc-300'}`}
                    >
                      {q}
                    </button>
                  ))}
                </div>
               </div>
            </div>
          </div>

          <div className="pt-6">
            <Button type="submit" className="w-full h-12 text-base shadow-none hover:shadow-lg transition-shadow">
              Launch Session
            </Button>
          </div>
        </form>
        
        <div className="mt-8 text-center">
            <p className="text-[10px] text-zinc-600 font-mono">
                v0.9.2-beta // RUNNING ON LOCAL CLIENT
            </p>
        </div>
      </div>
    </div>
  );
};