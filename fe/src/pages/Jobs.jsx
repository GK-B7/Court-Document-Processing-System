import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  MagnifyingGlassIcon,
  FunnelIcon,
  ArrowPathIcon,
  QueueListIcon,
  BoltIcon,
} from '@heroicons/react/24/outline';
import JobStatus from '../components/JobStatus';
import LoadingSpinner, { PageLoadingSpinner } from '../components/LoadingSpinner';
import { jobsAPI } from '../services/api';
import toast from 'react-hot-toast';
import clsx from 'clsx';

const statusFilters = [
  { label: 'All', value: 'all' },
  { label: 'Pending', value: 'pending' },
  { label: 'Processing', value: 'processing' },
  { label: 'Completed', value: 'completed' },
  { label: 'Review Required', value: 'review_required' },
  { label: 'Failed', value: 'failed' },
];

export default function Jobs() {
  const [jobs, setJobs] = useState([]);
  const [filteredJobs, setFilteredJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedStatus, setSelectedStatus] = useState('all');
  const navigate = useNavigate();

  useEffect(() => {
    fetchJobs();
    
    // Auto-refresh every 30 seconds for real-time updates
    const interval = setInterval(() => {
      fetchJobs(true); // Silent refresh
    }, 30000);
    
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    filterJobs();
  }, [jobs, searchTerm, selectedStatus]);

  const fetchJobs = async (silent = false) => {
    if (!silent) setLoading(true);
    setRefreshing(true);

    try {
      const response = await jobsAPI.getAllJobs({
        status: selectedStatus === 'all' ? null : selectedStatus,
        limit: 100
      });

      setJobs(response.data.jobs || []);
      if (!silent) setLoading(false);
    } catch (error) {
      console.error('Failed to fetch jobs:', error);
      if (!silent) {
        toast.error('Failed to load jobs');
        setLoading(false);
      }
    } finally {
      setRefreshing(false);
    }
  };

  const filterJobs = () => {
    let filtered = [...jobs];

    // Filter by status
    if (selectedStatus !== 'all') {
      filtered = filtered.filter(job => job.status === selectedStatus);
    }

    // Filter by search term
    if (searchTerm) {
      filtered = filtered.filter(job => 
        job.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
        job.job_id.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Sort by created_at descending (newest first)
    filtered.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

    setFilteredJobs(filtered);
  };

  const handleRefresh = () => {
    fetchJobs();
    toast.success('Jobs refreshed successfully!');
  };

  const handleStatusFilterChange = (newStatus) => {
    setSelectedStatus(newStatus);
    // Refetch jobs when status filter changes
    setTimeout(() => fetchJobs(), 100);
  };

  const getStatusCount = (status) => {
    if (status === 'all') return jobs.length;
    return jobs.filter(job => job.status === status).length;
  };

  // Get total actions executed across all jobs
  const getTotalActionsExecuted = () => {
    return jobs.reduce((total, job) => {
      const results = job.metadata || {};
      return total + (results.successful_executions || 0);
    }, 0);
  };

  if (loading) {
    return <PageLoadingSpinner text="Loading jobs..." />;
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header with Actions Summary */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">All Jobs</h1>
          <div className="flex items-center space-x-4 mt-1">
            <p className="text-gray-600">
              Monitor and manage document processing jobs
            </p>
            {getTotalActionsExecuted() > 0 && (
              <div className="flex items-center space-x-1 bg-green-100 text-green-800 px-2 py-1 rounded-md">
                <BoltIcon className="h-4 w-4" />
                <span className="text-sm font-medium">
                  {getTotalActionsExecuted()} actions executed
                </span>
              </div>
            )}
          </div>
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

      {/* Stats Overview */}
      <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
        {statusFilters.map((filter) => {
          const count = getStatusCount(filter.value);
          const isActive = selectedStatus === filter.value;
          
          return (
            <button
              key={filter.value}
              onClick={() => handleStatusFilterChange(filter.value)}
              className={clsx(
                'p-4 rounded-xl border text-left transition-all duration-200',
                isActive
                  ? 'border-primary-200 bg-primary-50 text-primary-700'
                  : 'border-gray-200 bg-white hover:bg-gray-50 text-gray-700'
              )}
            >
              <div className="text-2xl font-bold">{count}</div>
              <div className="text-sm">{filter.label}</div>
            </button>
          );
        })}
      </div>

      {/* Search and Filters */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0 sm:space-x-4">
          {/* Search */}
          <div className="relative flex-1 max-w-md">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <MagnifyingGlassIcon className="h-5 w-5 text-gray-400" />
            </div>
            <input
              type="text"
              placeholder="Search jobs by filename or ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
            />
          </div>

          {/* Status Filter */}
          <div className="flex items-center space-x-2">
            <FunnelIcon className="h-5 w-5 text-gray-400" />
            <select
              value={selectedStatus}
              onChange={(e) => handleStatusFilterChange(e.target.value)}
              className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              {statusFilters.map((filter) => (
                <option key={filter.value} value={filter.value}>
                  {filter.label} ({getStatusCount(filter.value)})
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Jobs List */}
      <div className="space-y-4">
        {filteredJobs.length === 0 ? (
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-12 text-center">
            <QueueListIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              {searchTerm || selectedStatus !== 'all' ? 'No jobs found' : 'No jobs yet'}
            </h3>
            <p className="text-gray-600 mb-6">
              {searchTerm || selectedStatus !== 'all' 
                ? 'Try adjusting your search or filter criteria'
                : 'Upload your first document to get started'
              }
            </p>
            {!searchTerm && selectedStatus === 'all' && (
              <button
                onClick={() => navigate('/upload')}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-lg text-white bg-primary-600 hover:bg-primary-700 transition-colors"
              >
                Upload Document
              </button>
            )}
          </div>
        ) : (
          <>
            {/* Results summary */}
            <div className="flex items-center justify-between text-sm text-gray-600">
              <span>
                Showing {filteredJobs.length} of {jobs.length} jobs
                {searchTerm && ` matching "${searchTerm}"`}
                {selectedStatus !== 'all' && ` with status "${statusFilters.find(f => f.value === selectedStatus)?.label}"`}
              </span>
              <span className="text-xs text-gray-500">
                Last updated: {new Date().toLocaleTimeString()}
              </span>
            </div>

            {/* Jobs grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {filteredJobs.map((job) => (
                <JobStatus
                  key={job.job_id}
                  job={{
                    ...job,
                    results: job.metadata // Map metadata to results for JobStatus component
                  }}
                  showDetails={true}
                  showProgress={true}
                  onClick={() => navigate(`/jobs/${job.job_id}`)}
                />
              ))}
            </div>
          </>
        )}
      </div>

      {/* Auto-refresh indicator */}
      {refreshing && (
        <div className="fixed bottom-4 right-4 bg-white rounded-lg shadow-lg border border-gray-200 p-3 flex items-center space-x-2">
          <LoadingSpinner size="sm" />
          <span className="text-sm text-gray-600">Refreshing jobs...</span>
        </div>
      )}
    </div>
  );
}
