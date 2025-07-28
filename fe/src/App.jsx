import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useParams, Link, useNavigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import toast from 'react-hot-toast'; // ✅ Add this import
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Upload from './pages/Upload';
import Jobs from './pages/Jobs';
import Review from './pages/Review';
import Settings from './pages/Settings';
import './index.css';

import {
  CheckCircleIcon,
  ExclamationTriangleIcon,
  BoltIcon, 
  XCircleIcon,
  LockClosedIcon,
  LockOpenIcon,
  BanknotesIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';

// ✅ Add clsx import
import clsx from 'clsx';

// Comprehensive JobDetails component with real data fetching and refresh button
const JobDetails = () => {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const [jobData, setJobData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false); // ✅ Add refreshing state
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchJobDetails();
    
    // Auto-refresh every 10 seconds if job is still processing
    const interval = setInterval(() => {
      if (jobData && (jobData.job.status === 'pending' || jobData.job.status === 'processing')) {
        fetchJobDetails(true);
      }
    }, 10000);
    
    return () => clearInterval(interval);
  }, [jobId]);

  const fetchJobDetails = async (silent = false) => {
    if (!silent) setLoading(true);
    
    try {
      // Import the API service dynamically
      const { jobsAPI } = await import('./services/api');
      const response = await jobsAPI.getJobById(jobId);
      setJobData(response.data);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch job details:', err);
      setError(err.response?.data?.detail || 'Failed to load job details');
    } finally {
      if (!silent) setLoading(false);
    }
  };

  // ✅ Add refresh handler
  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await fetchJobDetails();
      toast.success('Job details refreshed successfully!');
    } catch (err) {
      toast.error('Failed to refresh job details');
    } finally {
      setRefreshing(false);
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
  };

  const getStatusColor = (status) => {
    const colors = {
      pending: 'bg-gray-100 text-gray-800',
      processing: 'bg-blue-100 text-blue-800',
      completed: 'bg-green-100 text-green-800',
      review_required: 'bg-yellow-100 text-yellow-800',
      failed: 'bg-red-100 text-red-800',
    };
    return colors[status] || 'bg-gray-100 text-gray-800';
  };

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-300 rounded w-1/4 mb-6"></div>
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-8">
            <div className="h-6 bg-gray-300 rounded w-1/3 mb-4"></div>
            <div className="space-y-3">
              <div className="h-4 bg-gray-300 rounded w-3/4"></div>
              <div className="h-4 bg-gray-300 rounded w-1/2"></div>
              <div className="h-4 bg-gray-300 rounded w-2/3"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-8 text-center">
          <div className="text-red-500 text-6xl mb-4">⚠️</div>
          <h1 className="text-2xl font-bold text-gray-900 mb-4">Error Loading Job</h1>
          <p className="text-red-600 mb-6">{error}</p>
          <div className="space-x-4">
            <button
              onClick={() => fetchJobDetails()}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-lg text-white bg-primary-600 hover:bg-primary-700 transition-colors"
            >
              Try Again
            </button>
            <button
              onClick={() => navigate('/jobs')}
              className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-lg text-gray-700 bg-white hover:bg-gray-50 transition-colors"
            >
              Back to Jobs
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!jobData) {
    return (
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-8 text-center">
          <h1 className="text-2xl font-bold text-gray-900 mb-4">Job Not Found</h1>
          <p className="text-gray-600 mb-6">
            The job with ID {jobId} was not found.
          </p>
          <button
            onClick={() => navigate('/jobs')}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-lg text-white bg-primary-600 hover:bg-primary-700 transition-colors"
          >
            Back to Jobs
          </button>
        </div>
      </div>
    );
  }

  const { job, review_items = [], logs = [] } = jobData;

  return (
    <div className="max-w-6xl mx-auto space-y-6">   
      {/* ✅ Updated Header with Refresh Button */}
      <div className="flex items-center justify-between">
        <div>
          <button
            onClick={() => navigate('/jobs')}
            className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700 mb-2"
          >
            ← Back to Jobs
          </button>
          <h1 className="text-3xl font-bold text-gray-900">Job Details</h1>
          <p className="text-gray-600 mt-1">
            {job.filename || 'Document Processing Job'}
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          {/* ✅ Refresh Button */}
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

          <div className="text-right">
            <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(job.status)}`}>
              {job.status.replace('_', ' ').toUpperCase()}
            </div>
            <p className="text-sm text-gray-500 mt-1">
              ID: {job.job_id.slice(0, 8)}...
            </p>
          </div>
        </div>
      </div>

      {/* Job Overview */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Job Overview</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div>
            <p className="text-sm font-medium text-gray-500">Status</p>
            <p className="text-lg font-semibold text-gray-900 capitalize">
              {job.status.replace('_', ' ')}
            </p>
          </div>
          <div>
            <p className="text-sm font-medium text-gray-500">Progress</p>
            <p className="text-lg font-semibold text-gray-900">
              {Math.round((job.progress || 0) * 100)}%
            </p>
          </div>
          <div>
            <p className="text-sm font-medium text-gray-500">File Name</p>
            <p className="text-lg font-medium text-gray-900 truncate" title={job.filename}>
              {job.filename || 'N/A'}
            </p>
          </div>
          <div>
            <p className="text-sm font-medium text-gray-500">Created</p>
            <p className="text-lg font-medium text-gray-900">
              {formatDate(job.created_at)}
            </p>
          </div>
        </div>

        {/* Actions Summary*/}
        {(() => {
        const metadata = job.metadata || {};
        const actionsSummary = metadata.actions_summary || {};

        if (job.status === 'failed') {
          const errorMessage = job.error_message || metadata?.error_details || 'An unknown error occurred during processing.';
          return (
            <div className="bg-red-50 rounded-xl border border-red-200 shadow-sm p-6">
              <div className="flex items-start">
                <ExclamationTriangleIcon className="h-6 w-6 text-red-500 mr-3 flex-shrink-0" />
                <div>
                  <h2 className="text-xl font-semibold text-red-800 mb-2">Processing Failed</h2>
                  <p className="text-red-700">{errorMessage}</p>
                </div>
              </div>
            </div>
          );
        }

        const hasActions = actionsSummary.executed_actions?.length > 0 || 
                            actionsSummary.failed_actions?.length > 0 ||
                            actionsSummary.pending_reviews?.length > 0 ||
                            actionsSummary.reviewed_actions?.length > 0;
        
        if (!hasActions) {
            return (
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
                <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                <BoltIcon className="h-6 w-6 text-gray-400 mr-2" />
                Banking Actions
                </h2>
                <div className="text-center py-8">
                <BoltIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500">No banking actions were executed for this document</p>
                </div>
            </div>
            );
        }
        
        const getActionIcon = (actionName) => {
            const icons = {
            'freeze_funds': 'text-red-600',
            'release_funds': 'text-green-600', 
            'close_account': 'text-orange-600',
            'open_account': 'text-blue-600'
            };
            return icons[actionName] || 'text-gray-600';
        };
        
        return (
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-6 flex items-center">
                <BoltIcon className="h-6 w-6 text-indigo-500 mr-2" />
                Banking Actions Executed
            </h2>
            
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                <div className="flex items-center">
                    <CheckCircleIcon className="h-8 w-8 text-green-500 mr-3" />
                    <div>
                    <p className="text-2xl font-bold text-green-700">
                        {(actionsSummary.executed_actions?.length || 0) + (actionsSummary.reviewed_actions?.length || 0)}
                    </p>
                    <p className="text-sm text-green-600">Successfully Executed</p>
                    </div>
                </div>
                </div>
                
                <div className="bg-red-50 rounded-lg p-4 border border-red-200">
                <div className="flex items-center">
                    <XCircleIcon className="h-8 w-8 text-red-500 mr-3" />
                    <div>
                    <p className="text-2xl font-bold text-red-700">
                        {actionsSummary.failed_actions?.length || 0}
                    </p>
                    <p className="text-sm text-red-600">Failed</p>
                    </div>
                </div>
                </div>
                
                <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
                <div className="flex items-center">
                    <ExclamationTriangleIcon className="h-8 w-8 text-yellow-500 mr-3" />
                    <div>
                    <p className="text-2xl font-bold text-yellow-700">
                        {actionsSummary.pending_reviews?.length || 0}
                    </p>
                    <p className="text-sm text-yellow-600">Pending Review</p>
                    </div>
                </div>
                </div>
                
                <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                <div className="flex items-center">
                    <BoltIcon className="h-8 w-8 text-blue-500 mr-3" />
                    <div>
                    <p className="text-2xl font-bold text-blue-700">
                        {actionsSummary.total_actions_found || 0}
                    </p>
                    <p className="text-sm text-blue-600">Total Found</p>
                    </div>
                </div>
                </div>
            </div>

            {/* Detailed Actions List */}
            <div className="space-y-6">
                
                {/* Successfully Executed Actions */}
                {(actionsSummary.executed_actions?.length > 0 || actionsSummary.reviewed_actions?.length > 0) && (
                <div>
                    <h3 className="text-lg font-medium text-gray-900 mb-3 flex items-center">
                    <CheckCircleIcon className="h-5 w-5 text-green-500 mr-2" />
                    Successfully Executed Actions
                    </h3>
                    <div className="bg-green-50 rounded-lg p-4 space-y-3">
                    {[...(actionsSummary.executed_actions || []), ...(actionsSummary.reviewed_actions || [])].map((action, index) => (
                        <div key={index} className="flex items-center justify-between bg-white rounded-lg p-3 border border-green-200">
                        <div className="flex items-center space-x-3">
                            <div className={`p-2 rounded-lg bg-green-100 ${getActionIcon(action.action_name)}`}>
                            {action.action_name === 'freeze_funds' && <LockClosedIcon className="h-5 w-5" />}
                            {action.action_name === 'release_funds' && <LockOpenIcon className="h-5 w-5" />}
                            {action.action_name === 'close_account' && <XCircleIcon className="h-5 w-5" />}
                            {action.action_name === 'open_account' && <CheckCircleIcon className="h-5 w-5" />}
                            {!['freeze_funds', 'release_funds', 'close_account', 'open_account'].includes(action.action_name) && <BanknotesIcon className="h-5 w-5" />}
                            </div>
                            <div>
                            <p className="font-medium text-gray-900 capitalize">
                                {action.action_name.replace('_', ' ')}
                            </p>
                            <p className="text-sm text-gray-600">
                                Customer: {action.customer_id} | National ID: {action.national_id}
                            </p>
                            </div>
                        </div>
                        <div className="text-right">
                            <p className="text-sm font-medium text-green-600">Executed</p>
                            <p className="text-xs text-gray-500">
                            {new Date(action.executed_at).toLocaleString()}
                            </p>
                        </div>
                        </div>
                    ))}
                    </div>
                </div>
                )}

                {/* Failed Actions */}
                {actionsSummary.failed_actions?.length > 0 && (
                <div>
                    <h3 className="text-lg font-medium text-gray-900 mb-3 flex items-center">
                    <XCircleIcon className="h-5 w-5 text-red-500 mr-2" />
                    Failed Actions
                    </h3>
                    <div className="bg-red-50 rounded-lg p-4 space-y-3">
                    {actionsSummary.failed_actions.map((action, index) => (
                        <div key={index} className="flex items-center justify-between bg-white rounded-lg p-3 border border-red-200">
                        <div className="flex items-center space-x-3">
                            <div className={`p-2 rounded-lg bg-red-100 ${getActionIcon(action.action_name)}`}>
                            {action.action_name === 'freeze_funds' && <LockClosedIcon className="h-5 w-5" />}
                            {action.action_name === 'release_funds' && <LockOpenIcon className="h-5 w-5" />}
                            {action.action_name === 'close_account' && <XCircleIcon className="h-5 w-5" />}
                            {action.action_name === 'open_account' && <CheckCircleIcon className="h-5 w-5" />}
                            {!['freeze_funds', 'release_funds', 'close_account', 'open_account'].includes(action.action_name) && <BanknotesIcon className="h-5 w-5" />}
                            </div>
                            <div>
                            <p className="font-medium text-gray-900 capitalize">
                                {action.action_name.replace('_', ' ')}
                            </p>
                            <p className="text-sm text-gray-600">
                                Customer: {action.customer_id} | National ID: {action.national_id}
                            </p>
                            <p className="text-xs text-red-600">Error: {action.error}</p>
                            </div>
                        </div>
                        <div className="text-right">
                            <p className="text-sm font-medium text-red-600">Failed</p>
                            <p className="text-xs text-gray-500">
                            {new Date(action.attempted_at).toLocaleString()}
                            </p>
                        </div>
                        </div>
                    ))}
                    </div>
                </div>
                )}

                {/* Pending Review Actions */}
                {actionsSummary.pending_reviews?.length > 0 && (
                <div>
                    <h3 className="text-lg font-medium text-gray-900 mb-3 flex items-center">
                    <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500 mr-2" />
                    Pending Review Actions
                    </h3>
                    <div className="bg-yellow-50 rounded-lg p-4 space-y-3">
                    {actionsSummary.pending_reviews.map((action, index) => (
                        <div key={index} className="flex items-center justify-between bg-white rounded-lg p-3 border border-yellow-200">
                        <div className="flex items-center space-x-3">
                            <div className={`p-2 rounded-lg bg-yellow-100 ${getActionIcon(action.action_name)}`}>
                            {action.action_name === 'freeze_funds' && <LockClosedIcon className="h-5 w-5" />}
                            {action.action_name === 'release_funds' && <LockOpenIcon className="h-5 w-5" />}
                            {action.action_name === 'close_account' && <XCircleIcon className="h-5 w-5" />}
                            {action.action_name === 'open_account' && <CheckCircleIcon className="h-5 w-5" />}
                            {!['freeze_funds', 'release_funds', 'close_account', 'open_account'].includes(action.action_name) && <BanknotesIcon className="h-5 w-5" />}
                            </div>
                            <div>
                            <p className="font-medium text-gray-900 capitalize">
                                {action.action_name.replace('_', ' ')}
                            </p>
                            <p className="text-sm text-gray-600">
                                Customer: {action.customer_id} | National ID: {action.national_id}
                            </p>
                            <p className="text-xs text-yellow-600">
                                Confidence: {Math.round((action.confidence || 0) * 100)}%
                            </p>
                            </div>
                        </div>
                        <div className="text-right">
                            <p className="text-sm font-medium text-yellow-600">Pending Review</p>
                            <Link
                            to="/review"
                            className="text-xs text-blue-600 hover:text-blue-800 underline"
                            >
                            Go to Review Queue
                            </Link>
                        </div>
                        </div>
                    ))}
                    </div>
                </div>
                )}
                
            </div>
            </div>
        );
        })()}

        {/* Progress Bar */}
        <div className="mt-6">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium text-gray-700">Processing Progress</span>
            <span className="text-sm text-gray-600">{Math.round((job.progress || 0) * 100)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className={`h-2 rounded-full transition-all duration-500 ${
                job.status === 'completed' ? 'bg-green-500' :
                job.status === 'failed' ? 'bg-red-500' :
                job.status === 'review_required' ? 'bg-yellow-500' :
                'bg-blue-500'
              }`}
              style={{ width: `${Math.round((job.progress || 0) * 100)}%` }}
            />
          </div>
        </div>
      </div> 

      {/* Timestamps */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Timeline</h2>
        <div className="space-y-3">
          <div className="flex justify-between">
            <span className="text-gray-600">Created:</span>
            <span className="text-gray-900">{formatDate(job.created_at)}</span>
          </div>
          {job.started_at && (
            <div className="flex justify-between">
              <span className="text-gray-600">Started:</span>
              <span className="text-gray-900">{formatDate(job.started_at)}</span>
            </div>
          )}
          {job.completed_at && (
            <div className="flex justify-between">
              <span className="text-gray-600">Completed:</span>
              <span className="text-gray-900">{formatDate(job.completed_at)}</span>
            </div>
          )}
        </div>
      </div>

      {/* ✅ Refresh Indicator */}
      {refreshing && (
        <div className="fixed bottom-4 right-4 bg-white rounded-lg shadow-lg border border-gray-200 p-3 flex items-center space-x-2">
          <ArrowPathIcon className="h-4 w-4 animate-spin text-blue-500" />
          <span className="text-sm text-gray-600">Refreshing job details...</span>
        </div>
      )}
    </div>
  );
};

// NotFound component remains the same
const NotFound = () => {
  return (
    <div className="max-w-4xl mx-auto text-center">
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-12">
        <div className="text-6xl mb-4">🔍</div>
        <h1 className="text-3xl font-bold text-gray-900 mb-4">Page Not Found</h1>
        <p className="text-gray-600 mb-8">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <Link
          to="/"
          className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-lg text-white bg-primary-600 hover:bg-primary-700 transition-colors"
        >
          Go to Dashboard
        </Link>
      </div>
    </div>
  );
};

function App() {
  return (
    <Router>
      <div className="App min-h-screen bg-gray-50">
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/jobs" element={<Jobs />} />
            <Route path="/jobs/:jobId" element={<JobDetails />} />
            <Route path="/review" element={<Review />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Layout>
        
        {/* Toast notifications */}
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#fff',
              color: '#374151',
              boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
              border: '1px solid #e5e7eb',
              borderRadius: '12px',
              padding: '16px',
              maxWidth: '400px',
            },
            success: {
              iconTheme: {
                primary: '#22c55e',
                secondary: '#fff',
              },
              style: {
                borderLeft: '4px solid #22c55e',
              },
            },
            error: {
              iconTheme: {
                primary: '#ef4444',
                secondary: '#fff',
              },
              style: {
                borderLeft: '4px solid #ef4444',
              },
            },
            loading: {
              iconTheme: {
                primary: '#3b82f6',
                secondary: '#fff',
              },
              style: {
                borderLeft: '4px solid #3b82f6',
              },
            },
          }}
        />
      </div>
    </Router>
  );
}

export default App;
