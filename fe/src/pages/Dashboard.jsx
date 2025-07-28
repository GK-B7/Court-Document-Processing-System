import React, { useState, useEffect } from 'react';
import {
  DocumentTextIcon,
  QueueListIcon,
  ClipboardDocumentListIcon,
  ChartBarIcon,
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  XCircleIcon,
  BoltIcon,
} from '@heroicons/react/24/outline';
import StatsCard from '../components/StatsCard';
import JobStatus from '../components/JobStatus';
import LoadingSpinner, { PageLoadingSpinner } from '../components/LoadingSpinner';
import { Link, useNavigate } from 'react-router-dom';
import { systemAPI, jobsAPI } from '../services/api';
import toast from 'react-hot-toast';

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [recentJobs, setRecentJobs] = useState([]);
  const [systemHealth, setSystemHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    fetchDashboardData();
    
    // Refresh data every 30 seconds
    const interval = setInterval(fetchDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
        const [metricsRes, healthRes, jobsRes] = await Promise.all([
        systemAPI.getMetrics(),
        systemAPI.getHealth(),
        jobsAPI.getAllJobs({ limit: 5 })
        ]);

        setMetrics(metricsRes.data);
        setSystemHealth(healthRes.data);
        
        // Set recent jobs from actual API data
        const jobs = jobsRes.data.jobs || [];
        
        // Transform the jobs data with proper progress handling
        const transformedJobs = jobs.map(job => ({
        ...job,
        results: job.metadata, // Map metadata to results for JobStatus component compatibility
        // Fix progress calculation for completed jobs
        progress: (() => {
            // If job is completed, always show 100% progress
            if (job.status === 'completed') {
            return 1.0;
            }
            
            // If job failed, show 0% progress
            if (job.status === 'failed') {
            return 0.0;
            }
            
            // For other statuses, normalize the progress value
            if (typeof job.progress === 'number') {
            // If progress is already between 0-1, use as is
            if (job.progress <= 1) {
                return job.progress;
            }
            // If progress is percentage (0-100), convert to 0-1
            return job.progress / 100;
            }
            
            // Default progress based on status
            switch (job.status) {
            case 'pending':
                return 0.0;
            case 'processing':
                return 0.5;
            case 'review_required':
                return 0.8;
            default:
                return 0.0;
            }
        })()
        }));
        
        setRecentJobs(transformedJobs);
        setLoading(false);
    } catch (error) {
        console.error('Failed to fetch dashboard data:', error);
        toast.error('Failed to load dashboard data');
        setLoading(false);
    }
    };

  // Get action display for jobs
  const getJobActionsDisplay = (job) => {
    const metadata = job.metadata || {};
    const actionsSummary = metadata.actions_summary || {};
    const actions = [];

    // Get executed actions with specific names
    const executedActions = actionsSummary.executed_actions || [];
    const reviewedActions = actionsSummary.reviewed_actions || [];
    
    const allExecuted = [...executedActions, ...reviewedActions];
    
    if (allExecuted.length > 0) {
        const actionNames = allExecuted.map(action => 
        action.action_name.replace('_', ' ')
        ).join(', ');
        actions.push(`${allExecuted.length} executed: ${actionNames}`);
    }
    
    const failedActions = actionsSummary.failed_actions || [];
    if (failedActions.length > 0) {
        actions.push(`${failedActions.length} failed`);
    }
    
    const pendingReviews = actionsSummary.pending_reviews || [];
    if (pendingReviews.length > 0) {
        const actionNames = pendingReviews.map(action => 
        action.action_name.replace('_', ' ')
        ).join(', ');
        actions.push(`${pendingReviews.length} pending: ${actionNames}`);
    }

    return actions.length > 0 ? actions.join(' | ') : 'No banking actions found';
    };

  // Get total actions executed across all jobs
  const getTotalActionsExecuted = () => {
    return recentJobs.reduce((total, job) => {
      const results = job.metadata || {};
      return total + (results.successful_executions || 0);
    }, 0);
  };

  if (loading) {
    return <PageLoadingSpinner text="Loading dashboard..." />;
  }

  const calculateSuccessRate = () => {
    if (!metrics || metrics.total_jobs === 0) return 0;
    return Math.round((metrics.successful_jobs / metrics.total_jobs) * 100);
  };

  const getHealthStatus = () => {
    if (!systemHealth) return { color: 'gray', text: 'Unknown' };
    
    if (systemHealth.status === 'healthy') {
      return { color: 'success', text: 'Healthy' };
    } else {
      return { color: 'danger', text: 'Issues Detected' };
    }
  };

  const healthStatus = getHealthStatus();

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Welcome Header with Actions Summary */}
      <div className="bg-gradient-to-r from-primary-600 to-primary-800 rounded-xl text-white p-6">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">Welcome to Document Processing</h1>
            <p className="text-primary-100">
              Process banking documents automatically with AI-powered extraction and action matching.
            </p>
            <div className="mt-4">
              <Link
                to="/upload"
                className="inline-flex items-center px-4 py-2 bg-white text-primary-600 rounded-lg font-medium hover:bg-primary-50 transition-colors"
              >
                <DocumentTextIcon className="h-5 w-5 mr-2" />
                Upload New Document
              </Link>
            </div>
          </div>
          
          {/* Actions Summary */}
          {getTotalActionsExecuted() > 0 && (
            <div className="bg-white bg-opacity-20 rounded-lg p-4 text-center">
              <div className="flex items-center justify-center mb-1">
                <BoltIcon className="h-6 w-6 text-primary-100 mr-2" />
                <span className="text-2xl font-bold text-white">
                  {getTotalActionsExecuted()}
                </span>
              </div>
              <p className="text-primary-100 text-sm">Actions Executed Today</p>
            </div>
          )}
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Total Jobs"
          value={metrics?.total_jobs || 0}
          subtitle="All time"
          icon={QueueListIcon}
          color="primary"
        />
        
        <StatsCard
          title="Successful Jobs"
          value={metrics?.successful_jobs || 0}
          subtitle={`${calculateSuccessRate()}% success rate`}
          trend="up"
          trendValue={`${calculateSuccessRate()}%`}
          icon={CheckCircleIcon}
          color="success"
        />
        
        <StatsCard
          title="Pending Reviews"
          value={metrics?.jobs_in_review || 0}
          subtitle="Needs attention"
          icon={ClipboardDocumentListIcon}
          color="warning"
        />
        
        <StatsCard
          title="Failed Jobs"
          value={metrics?.failed_jobs || 0}
          subtitle="Requires investigation"
          icon={XCircleIcon}
          color="danger"
        />
      </div>

      {/* Secondary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatsCard
          title="Processing Time"
          value={`${Math.round(metrics?.average_processing_time || 0)}s`}
          subtitle="Average per document"
          icon={ClockIcon}
          color="gray"
        />
        
        <StatsCard
          title="Queue Size"
          value={metrics?.current_queue_size || 0}
          subtitle="Jobs in queue"
          icon={ChartBarIcon}
          color="primary"
        />
        
        <StatsCard
          title="System Status"
          value={healthStatus.text}
          subtitle={systemHealth?.customers_count ? `${systemHealth.customers_count} customers loaded` : 'Status unknown'}
          icon={systemHealth?.status === 'healthy' ? CheckCircleIcon : ExclamationTriangleIcon}
          color={healthStatus.color}
        />
      </div>

      {/* Recent Jobs and System Info */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Jobs */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
            <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-900">Recent Jobs</h2>
              <Link
                to="/jobs"
                className="text-primary-600 hover:text-primary-700 text-sm font-medium"
              >
                View All →
              </Link>
            </div>
            <div className="p-6">
              {recentJobs.length === 0 ? (
                <div className="text-center py-8">
                  <QueueListIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-500">No recent jobs</p>
                  <Link
                    to="/upload"
                    className="text-primary-600 hover:text-primary-700 text-sm font-medium mt-2 inline-block"
                  >
                    Upload your first document
                  </Link>
                </div>
              ) : (
                <div className="space-y-4">
                  {recentJobs.map((job) => (
                    <div key={job.job_id}>
                      <JobStatus
                        job={job}
                        showDetails={false}
                        showProgress={true}
                        className="border-0 shadow-none bg-gray-50"
                        onClick={() => navigate(`/jobs/${job.job_id}`)}
                      />
                      {/* Action summary below each job */}
                      <div className="mt-2 ml-12 flex items-center space-x-2">
                        <BoltIcon className="h-3 w-3 text-indigo-500" />
                        <span className="text-xs text-indigo-600 font-medium">
                          {getJobActionsDisplay(job)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* System Information */}
        <div className="space-y-6">
          {/* Quick Actions */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
            <div className="space-y-3">
              <Link
                to="/upload"
                className="flex items-center p-3 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors"
              >
                <DocumentTextIcon className="h-5 w-5 text-primary-600 mr-3" />
                <span className="text-sm font-medium text-gray-900">Upload Document</span>
              </Link>
              
              <Link
                to="/review"
                className="flex items-center p-3 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors"
              >
                <ClipboardDocumentListIcon className="h-5 w-5 text-warning-600 mr-3" />
                <div className="flex-1">
                  <span className="text-sm font-medium text-gray-900">Review Queue</span>
                  {metrics?.jobs_in_review > 0 && (
                    <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-warning-100 text-warning-800">
                      {metrics.jobs_in_review} pending
                    </span>
                  )}
                </div>
              </Link>
              
              <Link
                to="/jobs"
                className="flex items-center p-3 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors"
              >
                <QueueListIcon className="h-5 w-5 text-gray-600 mr-3" />
                <span className="text-sm font-medium text-gray-900">All Jobs</span>
              </Link>
            </div>
          </div>

          {/* System Health */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">System Health</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">API Status</span>
                <div className="flex items-center">
                  <div className={`h-2 w-2 rounded-full mr-2 ${
                    systemHealth?.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                  }`} />
                  <span className="text-sm font-medium text-gray-900">
                    {systemHealth?.status || 'Unknown'}
                  </span>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Database</span>
                <span className="text-sm font-medium text-gray-900">
                  {systemHealth?.database || 'Unknown'}
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Customers</span>
                <span className="text-sm font-medium text-gray-900">
                  {systemHealth?.customers_count || 0} loaded
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Actions</span>
                <span className="text-sm font-medium text-gray-900">
                  {systemHealth?.actions_count || 0} available
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
