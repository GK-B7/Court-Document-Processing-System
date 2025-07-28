import React, { useState, useEffect } from 'react';
import {
  Cog6ToothIcon,
  ServerIcon,
  CircleStackIcon,
  UserGroupIcon,
  DocumentTextIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ArrowPathIcon,
  InformationCircleIcon,
  TrashIcon,
} from '@heroicons/react/24/outline';
import StatsCard from '../components/StatsCard';
import LoadingSpinner, { PageLoadingSpinner } from '../components/LoadingSpinner';
import { systemAPI } from '../services/api';
import toast from 'react-hot-toast';
import clsx from 'clsx';

export default function Settings() {
  const [systemHealth, setSystemHealth] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [actions, setActions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    fetchSystemData();
    
    // Auto-refresh every 60 seconds
    const interval = setInterval(() => {
      fetchSystemData(true);
    }, 60000);
    
    return () => clearInterval(interval);
  }, []);

  const fetchSystemData = async (silent = false) => {
    if (!silent) setLoading(true);
    setRefreshing(true);

    try {
      const [healthRes, metricsRes, actionsRes] = await Promise.all([
        systemAPI.getHealth(),
        systemAPI.getMetrics(),
        systemAPI.getActions()
      ]);

      setSystemHealth(healthRes.data);
      setMetrics(metricsRes.data);
      setActions(actionsRes.data.actions || []);
      
      if (!silent) setLoading(false);
    } catch (error) {
      console.error('Failed to fetch system data:', error);
      if (!silent) {
        toast.error('Failed to load system data');
        setLoading(false);
      }
    } finally {
      setRefreshing(false);
    }
  };

  const handleRefresh = () => {
    fetchSystemData();
    toast.success('System data refreshed');
  };

  const handleTestConnection = async () => {
    try {
      const response = await systemAPI.getHealth();
      if (response.data.status === 'healthy') {
        toast.success('Database connection successful!');
      } else {
        toast.error('Database connection issues detected');
      }
    } catch (error) {
      toast.error('Failed to test database connection');
    }
  };

  const formatUptime = (timestamp) => {
    if (!timestamp) return 'Unknown';
    const startTime = new Date(timestamp);
    const now = new Date();
    const uptimeMs = now - startTime;
    const hours = Math.floor(uptimeMs / (1000 * 60 * 60));
    const minutes = Math.floor((uptimeMs % (1000 * 60 * 60)) / (1000 * 60));
    return `${hours}h ${minutes}m`;
  };

  if (loading) {
    return <PageLoadingSpinner text="Loading system settings..." />;
  }

  const isSystemHealthy = systemHealth?.status === 'healthy';

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">System Settings</h1>
          <p className="text-gray-600 mt-1">
            Monitor system health and manage configuration
          </p>
        </div>
        
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-lg text-gray-700 bg-white hover:bg-gray-50 focus:ring-2 focus:ring-primary-500 disabled:opacity-50 transition-colors"
        >
          <ArrowPathIcon className={clsx(
            'h-4 w-4 mr-2',
            refreshing && 'animate-spin'
          )} />
          {refreshing ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {/* System Health Alert */}
      {!isSystemHealthy && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4">
          <div className="flex items-center">
            <ExclamationTriangleIcon className="h-5 w-5 text-red-500 mr-2" />
            <h3 className="text-red-800 font-medium">System Health Issues Detected</h3>
          </div>
          <p className="text-red-700 text-sm mt-1">
            Some system components may not be functioning properly. Check the details below.
          </p>
        </div>
      )}

      {/* System Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <StatsCard
          title="System Status"
          value={isSystemHealthy ? "Healthy" : "Issues"}
          subtitle={systemHealth?.database || "Unknown"}
          icon={isSystemHealthy ? CheckCircleIcon : ExclamationTriangleIcon}
          color={isSystemHealthy ? "success" : "danger"}
        />
        
        <StatsCard
          title="Database"
          value={systemHealth?.database || "Unknown"}
          subtitle={`${systemHealth?.customers_count || 0} customers`}
          icon={CircleStackIcon}
          color="primary"
        />
        
        <StatsCard
          title="Total Jobs"
          value={metrics?.total_jobs || 0}
          subtitle={`${metrics?.successful_jobs || 0} successful`}
          icon={DocumentTextIcon}
          color="gray"
        />
        
        <StatsCard
          title="Processing Time"
          value={`${Math.round(metrics?.average_processing_time || 0)}s`}
          subtitle="Average per document"
          icon={Cog6ToothIcon}
          color="warning"
        />
      </div>

      {/* System Information */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* System Health Details */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900 flex items-center">
              <ServerIcon className="h-5 w-5 mr-2" />
              System Health
            </h2>
          </div>
          <div className="p-6 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">API Status</span>
              <div className="flex items-center">
                <div className={clsx(
                  'h-2 w-2 rounded-full mr-2',
                  isSystemHealthy ? 'bg-green-500' : 'bg-red-500'
                )} />
                <span className="text-sm font-medium text-gray-900">
                  {systemHealth?.status || 'Unknown'}
                </span>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Database Connection</span>
              <span className="text-sm font-medium text-gray-900">
                {systemHealth?.database || 'Unknown'}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Uptime</span>
              <span className="text-sm font-medium text-gray-900">
                {formatUptime(systemHealth?.timestamp)}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Job Manager</span>
              <span className="text-sm font-medium text-gray-900">
                {systemHealth?.job_manager || 'Unknown'}
              </span>
            </div>

            <div className="pt-4 border-t border-gray-200">
              <button
                onClick={handleTestConnection}
                className="w-full inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-lg text-white bg-primary-600 hover:bg-primary-700 transition-colors"
              >
                <CircleStackIcon className="h-4 w-4 mr-2" />
                Test Database Connection
              </button>
            </div>
          </div>
        </div>

        {/* Processing Statistics */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900 flex items-center">
              <DocumentTextIcon className="h-5 w-5 mr-2" />
              Processing Statistics
            </h2>
          </div>
          <div className="p-6 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Total Jobs Processed</span>
              <span className="text-sm font-medium text-gray-900">
                {metrics?.total_jobs || 0}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Successful Jobs</span>
              <span className="text-sm font-medium text-green-600">
                {metrics?.successful_jobs || 0}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Failed Jobs</span>
              <span className="text-sm font-medium text-red-600">
                {metrics?.failed_jobs || 0}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Jobs in Review</span>
              <span className="text-sm font-medium text-warning-600">
                {metrics?.jobs_in_review || 0}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Success Rate</span>
              <span className="text-sm font-medium text-gray-900">
                {metrics?.total_jobs > 0 
                  ? `${Math.round((metrics.successful_jobs / metrics.total_jobs) * 100)}%`
                  : '0%'
                }
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Database Information */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900 flex items-center">
            <CircleStackIcon className="h-5 w-5 mr-2" />
            Database Information
          </h2>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <UserGroupIcon className="h-8 w-8 text-blue-500 mx-auto mb-2" />
              <div className="text-2xl font-bold text-gray-900">
                {systemHealth?.customers_count || 0}
              </div>
              <div className="text-sm text-gray-600">Customers Loaded</div>
            </div>
            
            <div className="text-center">
              <Cog6ToothIcon className="h-8 w-8 text-green-500 mx-auto mb-2" />
              <div className="text-2xl font-bold text-gray-900">
                {systemHealth?.actions_count || 0}
              </div>
              <div className="text-sm text-gray-600">Available Actions</div>
            </div>
            
            <div className="text-center">
              <DocumentTextIcon className="h-8 w-8 text-purple-500 mx-auto mb-2" />
              <div className="text-2xl font-bold text-gray-900">
                {systemHealth?.jobs_count || 0}
              </div>
              <div className="text-sm text-gray-600">Total Jobs</div>
            </div>
          </div>
        </div>
      </div>

      {/* Supported Actions */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Supported Actions</h2>
          <p className="text-sm text-gray-600 mt-1">
            Actions that can be extracted and executed from documents
          </p>
        </div>
        <div className="p-6">
          {actions.length === 0 ? (
            <div className="text-center py-8">
              <ExclamationTriangleIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">No actions configured</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {actions.map((action, index) => (
                <div
                  key={index}
                  className="border border-gray-200 rounded-lg p-4"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-medium text-gray-900">
                        {action.action_name}
                      </h3>
                      <p className="text-sm text-gray-600 mt-1">
                        {action.description}
                      </p>
                    </div>
                    <CheckCircleIcon className="h-5 w-5 text-green-500" />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* System Information Note */}
      <div className="bg-blue-50 rounded-xl p-6">
        <div className="flex items-start">
          <InformationCircleIcon className="h-6 w-6 text-blue-500 mt-0.5 mr-3 flex-shrink-0" />
          <div>
            <h3 className="font-semibold text-blue-900 mb-2">System Information</h3>
            <div className="text-sm text-blue-800 space-y-1">
              <p>• This system processes banking documents to extract customer actions automatically</p>
              <p>• Supported file types: PDF (up to 10MB)</p>
              <p>• Actions requiring low confidence are routed to human review</p>
              <p>• All processing activities are logged for audit purposes</p>
              <p>• System data refreshes automatically every 60 seconds</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
