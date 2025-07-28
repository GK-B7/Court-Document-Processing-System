import React from 'react';
import clsx from 'clsx';

export default function LoadingSpinner({ 
  size = 'md', 
  color = 'primary',
  className = '',
  text = null,
  overlay = false,
}) {
  const sizeClasses = {
    xs: 'h-3 w-3',
    sm: 'h-4 w-4',
    md: 'h-6 w-6',
    lg: 'h-8 w-8',
    xl: 'h-12 w-12',
    '2xl': 'h-16 w-16',
  };

  const colorClasses = {
    primary: 'text-primary-600',
    white: 'text-white',
    gray: 'text-gray-600',
    success: 'text-success-600',
    warning: 'text-warning-600',
    danger: 'text-danger-600',
  };

  const textSizeClasses = {
    xs: 'text-xs',
    sm: 'text-sm',
    md: 'text-sm',
    lg: 'text-base',
    xl: 'text-lg',
    '2xl': 'text-xl',
  };

  const SpinnerIcon = () => (
    <svg
      className={clsx(
        'animate-spin',
        sizeClasses[size],
        colorClasses[color],
        className
      )}
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );

  // Overlay version
  if (overlay) {
    return (
      <div className="fixed inset-0 bg-gray-900 bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-xl p-8 shadow-xl flex flex-col items-center space-y-4">
          <SpinnerIcon />
          {text && (
            <p className={clsx(
              'font-medium text-gray-900',
              textSizeClasses[size]
            )}>
              {text}
            </p>
          )}
        </div>
      </div>
    );
  }

  // Inline version with text
  if (text) {
    return (
      <div className={clsx('flex items-center justify-center space-x-3', className)}>
        <SpinnerIcon />
        <span className={clsx(
          'font-medium',
          colorClasses[color] === 'text-white' ? 'text-white' : 'text-gray-900',
          textSizeClasses[size]
        )}>
          {text}
        </span>
      </div>
    );
  }

  // Simple spinner
  return <SpinnerIcon />;
}

// Pre-built spinner variations
export const PageLoadingSpinner = ({ text = "Loading..." }) => (
  <div className="flex items-center justify-center min-h-[400px]">
    <LoadingSpinner size="xl" text={text} />
  </div>
);

export const ButtonLoadingSpinner = ({ size = 'sm' }) => (
  <LoadingSpinner size={size} color="white" />
);

export const OverlayLoadingSpinner = ({ text = "Processing..." }) => (
  <LoadingSpinner overlay text={text} size="lg" />
);

export const InlineLoadingSpinner = ({ text, size = 'md' }) => (
  <LoadingSpinner text={text} size={size} />
);
