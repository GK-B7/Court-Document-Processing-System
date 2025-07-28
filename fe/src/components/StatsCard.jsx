import React from 'react';
import {
  ArrowUpIcon,
  ArrowDownIcon,
  MinusIcon,
} from '@heroicons/react/24/solid';
import clsx from 'clsx';

export default function StatsCard({
  title,
  value,
  subtitle,
  trend,
  trendValue,
  icon: Icon,
  color = 'primary',
  loading = false,
  className = '',
}) {
  const colorClasses = {
    primary: {
      bg: 'bg-primary-50',
      icon: 'text-primary-600',
      value: 'text-primary-900',
    },
    success: {
      bg: 'bg-success-50',
      icon: 'text-success-600',
      value: 'text-success-900',
    },
    warning: {
      bg: 'bg-warning-50',
      icon: 'text-warning-600',
      value: 'text-warning-900',
    },
    danger: {
      bg: 'bg-danger-50',
      icon: 'text-danger-600',
      value: 'text-danger-900',
    },
    gray: {
      bg: 'bg-gray-50',
      icon: 'text-gray-600',
      value: 'text-gray-900',
    },
  };

  const colors = colorClasses[color] || colorClasses.primary;

  const getTrendIcon = () => {
    if (!trend || trend === 'neutral') return MinusIcon;
    return trend === 'up' ? ArrowUpIcon : ArrowDownIcon;
  };

  const getTrendColor = () => {
    if (!trend || trend === 'neutral') return 'text-gray-500';
    return trend === 'up' ? 'text-green-600' : 'text-red-600';
  };

  const TrendIcon = getTrendIcon();

  if (loading) {
    return (
      <div className={clsx(
        'bg-white rounded-xl border border-gray-200 shadow-sm p-6',
        'animate-pulse',
        className
      )}>
        <div className="flex items-center justify-between">
          <div className="space-y-3 flex-1">
            <div className="h-4 bg-gray-200 rounded w-3/4"></div>
            <div className="h-8 bg-gray-200 rounded w-1/2"></div>
            <div className="h-3 bg-gray-200 rounded w-1/3"></div>
          </div>
          <div className="h-12 w-12 bg-gray-200 rounded-lg"></div>
        </div>
      </div>
    );
  }

  return (
    <div className={clsx(
      'bg-white rounded-xl border border-gray-200 shadow-sm p-6 hover:shadow-md transition-shadow duration-200',
      className
    )}>
      <div className="flex items-center justify-between">
        <div className="flex-1">
          {/* Title */}
          <p className="text-sm font-medium text-gray-600 mb-2">{title}</p>
          
          {/* Value */}
          <div className="flex items-baseline space-x-2">
            <p className={clsx(
              'text-3xl font-bold',
              colors.value
            )}>
              {typeof value === 'number' ? value.toLocaleString() : value}
            </p>
            
            {/* Trend */}
            {trend && trendValue && (
              <div className={clsx(
                'flex items-center text-sm font-medium',
                getTrendColor()
              )}>
                <TrendIcon className="h-3 w-3 mr-1" />
                {trendValue}
              </div>
            )}
          </div>

          {/* Subtitle */}
          {subtitle && (
            <p className="text-sm text-gray-500 mt-1">{subtitle}</p>
          )}
        </div>

        {/* Icon */}
        {Icon && (
          <div className={clsx(
            'p-3 rounded-lg',
            colors.bg
          )}>
            <Icon className={clsx('h-6 w-6', colors.icon)} />
          </div>
        )}
      </div>
    </div>
  );
}
