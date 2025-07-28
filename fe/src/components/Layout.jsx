import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  HomeIcon,
  CloudArrowUpIcon,
  QueueListIcon,
  ClipboardDocumentListIcon,
  Cog6ToothIcon,
  Bars3Icon,
  XMarkIcon,
  DocumentTextIcon,
  BellIcon,
} from '@heroicons/react/24/outline';
import clsx from 'clsx';

const navigation = [
  { name: 'Dashboard', href: '/', icon: HomeIcon },
  { name: 'Upload Document', href: '/upload', icon: CloudArrowUpIcon },
  { name: 'All Jobs', href: '/jobs', icon: QueueListIcon },
  { name: 'Review Queue', href: '/review', icon: ClipboardDocumentListIcon },
  { name: 'Settings', href: '/settings', icon: Cog6ToothIcon },
];

export default function Layout({ children }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Mobile sidebar */}
      <div className={clsx(
        'fixed inset-0 z-50 lg:hidden',
        sidebarOpen ? 'block' : 'hidden'
      )}>
        <div className="fixed inset-0 bg-gray-600 bg-opacity-75" onClick={() => setSidebarOpen(false)} />
        <div className="fixed inset-y-0 left-0 flex w-64 flex-col bg-white shadow-xl">
          <div className="flex h-16 items-center justify-between px-4 border-b border-gray-200">
            <div className="flex items-center">
              <DocumentTextIcon className="h-8 w-8 text-primary-600" />
              <span className="ml-2 text-xl font-bold text-gray-900">DocProcessor</span>
            </div>
            <button
              onClick={() => setSidebarOpen(false)}
              className="rounded-md p-2 text-gray-400 hover:text-gray-500"
            >
              <XMarkIcon className="h-6 w-6" />
            </button>
          </div>
          <nav className="flex-1 px-4 py-4 space-y-1">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  onClick={() => setSidebarOpen(false)}
                  className={clsx(
                    'flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-colors duration-200',
                    isActive
                      ? 'bg-primary-100 text-primary-700 border-r-2 border-primary-600'
                      : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                  )}
                >
                  <item.icon className="mr-3 h-5 w-5" />
                  {item.name}
                </Link>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Desktop sidebar */}
      <div className="hidden lg:fixed lg:inset-y-0 lg:flex lg:w-64 lg:flex-col">
        <div className="flex flex-col flex-grow bg-white border-r border-gray-200 shadow-sm">
          {/* Logo */}
          <div className="flex h-16 items-center px-6 border-b border-gray-200">
            <DocumentTextIcon className="h-8 w-8 text-primary-600" />
            <span className="ml-2 text-xl font-bold text-gray-900">DocProcessor</span>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4 py-4 space-y-1">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={clsx(
                    'flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-colors duration-200',
                    isActive
                      ? 'bg-primary-100 text-primary-700 border-r-2 border-primary-600'
                      : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                  )}
                >
                  <item.icon className="mr-3 h-5 w-5" />
                  {item.name}
                </Link>
              );
            })}
          </nav>

          {/* Footer info */}
          <div className="px-4 py-4 border-t border-gray-200">
            <div className="text-xs text-gray-500">
              <div className="font-medium">Banking Document Processor</div>
              <div>v1.0.0</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="lg:pl-64">
        {/* Top header */}
        <header className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-40">
          <div className="flex h-16 items-center justify-between px-4 sm:px-6 lg:px-8">
            <div className="flex items-center">
              <button
                onClick={() => setSidebarOpen(true)}
                className="lg:hidden rounded-md p-2 text-gray-400 hover:text-gray-500"
              >
                <Bars3Icon className="h-6 w-6" />
              </button>
              <h1 className="ml-2 lg:ml-0 text-2xl font-semibold text-gray-900">
                {navigation.find(item => item.href === location.pathname)?.name || 'Document Processing'}
              </h1>
            </div>

            <div className="flex items-center space-x-4">
              {/* Notifications */}
              <button className="rounded-full p-2 text-gray-400 hover:text-gray-500 relative">
                <BellIcon className="h-6 w-6" />
                <span className="absolute top-1 right-1 h-2 w-2 bg-red-500 rounded-full"></span>
              </button>

              {/* System status */}
              <div className="flex items-center space-x-2">
                <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm text-gray-600 hidden sm:block">System Online</span>
              </div>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1">
          <div className="py-6 px-4 sm:px-6 lg:px-8">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
