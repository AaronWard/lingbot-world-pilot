import React from 'react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  isLoading?: boolean;
}

export const Button: React.FC<ButtonProps> = ({ 
  children, 
  variant = 'primary', 
  isLoading = false, 
  className = '', 
  disabled,
  ...props 
}) => {
  const baseStyles = "relative inline-flex items-center justify-center px-6 py-2.5 text-sm font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-black disabled:opacity-50 disabled:cursor-not-allowed tracking-tight rounded-md";
  
  const variants = {
    primary: "bg-white text-black hover:bg-zinc-200 border border-transparent focus:ring-white shadow-[0_0_10px_rgba(255,255,255,0.1)]",
    secondary: "bg-black text-zinc-300 border border-zinc-800 hover:border-zinc-600 hover:text-white focus:ring-zinc-700",
    danger: "bg-transparent text-red-500 border border-red-900/30 hover:bg-red-950/30 hover:border-red-800 focus:ring-red-900",
    ghost: "bg-transparent text-zinc-500 hover:text-white"
  };

  return (
    <button 
      className={`${baseStyles} ${variants[variant]} ${className}`}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading && (
        <span className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
        </span>
      )}
      <span className={isLoading ? 'opacity-0' : ''}>{children}</span>
    </button>
  );
};