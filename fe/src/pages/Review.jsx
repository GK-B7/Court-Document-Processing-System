import React, { useState, useEffect } from 'react';
import {
  EyeIcon,
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';
import ReviewItem from '../components/ReviewItem';
import StatsCard from '../components/StatsCard';
import LoadingSpinner, { PageLoadingSpinner } from '../components/LoadingSpinner';
import { reviewAPI } from '../services/api';
import toast from 'react-hot-toast';
import clsx from 'clsx';

const priorityFilters = [
  { label: 'All Priorities', value: 'all' },
  { label: 'Critical', value: 'critical' },
  { label: 'High', value: 'high' },
  { label: 'Medium', value: 'medium' },
  { label: 'Low', value: 'low' },
];

export default function Review() {
  const [reviewItems, setReviewItems] = useState([]);
  const [filteredItems, setFilteredItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedPriority, setSelectedPriority] = useState('all');
  const [processingItems, setProcessingItems] = useState(new Set());

  useEffect(() => {
    fetchReviewQueue();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(() => {
      fetchReviewQueue(true); // Silent refresh
    }, 30000);
    
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    filterItems();
  }, [reviewItems, selectedPriority]);

  const fetchReviewQueue = async (silent = false) => {
    if (!silent) setLoading(true);
    setRefreshing(true);

    try {
      const response = await reviewAPI.getReviewQueue();
      
      // Transform backend data to match frontend expectations
      const items = response.data.map(item => ({
        ...item,
        review_reasons: item.review_reason ? [item.review_reason] : [],
        risk_level: 'medium', // Default since backend doesn't provide this
        risk_factors: [],
        similarity_score: item.confidence || 0,
        customer_name: item.customer_id,
        customer_status: 'active',
        customer_found: !!item.customer_id,
        priority: item.priority || 5,
      }));

      setReviewItems(items);
      if (!silent) setLoading(false);
    } catch (error) {
      console.error('Failed to fetch review queue:', error);
      if (!silent) {
        toast.error('Failed to load review queue');
        setLoading(false);
      }
    } finally {
      setRefreshing(false);
    }
  };

  const filterItems = () => {
    let filtered = [...reviewItems];

    // Filter by priority
    if (selectedPriority !== 'all') {
      filtered = filtered.filter(item => 
        getPriorityFromNumber(item.priority) === selectedPriority
      );
    }

    // Sort by priority (critical first) then by creation date
    filtered.sort((a, b) => {
      const priorityOrder = { 'critical': 1, 'high': 2, 'medium': 3, 'low': 4 };
      const aPriority = getPriorityFromNumber(a.priority);
      const bPriority = getPriorityFromNumber(b.priority);
      
      if (priorityOrder[aPriority] !== priorityOrder[bPriority]) {
        return priorityOrder[aPriority] - priorityOrder[bPriority];
      }
      
      return new Date(b.created_at) - new Date(a.created_at);
    });

    setFilteredItems(filtered);
  };

  const getPriorityFromNumber = (priorityNum) => {
    const priorityMap = {
      1: 'critical',
      2: 'high',
      3: 'medium',
      4: 'low',
      5: 'low'
    };
    return priorityMap[priorityNum] || 'medium';
  };

  const getPriorityCount = (priority) => {
    if (priority === 'all') return reviewItems.length;
    return reviewItems.filter(item => 
      getPriorityFromNumber(item.priority) === priority
    ).length;
  };

    const handleApprove = async (reviewId, data) => {
    setProcessingItems(prev => new Set([...prev, reviewId]));
    
    try {
        const response = await reviewAPI.submitReview(reviewId, data);
        
        // Remove item from local state
        setReviewItems(prev => prev.filter(item => item.id !== reviewId));
        
        // Show detailed success message with execution status
        if (response.data.execution_result) {
        const execResult = response.data.execution_result;
        if (execResult.status === 'success') {
            toast.success(
            `✅ Review approved! Action "${execResult.action_name.replace('_', ' ')}" executed successfully for customer ${execResult.customer_id}`,
            { duration: 6000 }
            );
        } else {
            toast.error(
            `⚠️ Review approved but action execution failed: ${execResult.error}`,
            { duration: 6000 }
            );
        }
        } else {
        toast.success('Review approved successfully!');
        }
        
        // If no more reviews for this job, show completion message
        if (response.data.remaining_reviews === 0) {
        toast.success(
            `🎉 Job ${response.data.job_id.slice(0, 8)}... completed! All reviews processed.`,
            { duration: 6000 }
        );
        }
        
    } catch (error) {
        console.error('Failed to approve review:', error);
        toast.error('Failed to approve review: ' + (error.response?.data?.detail || error.message));
    } finally {
        setProcessingItems(prev => {
        const newSet = new Set(prev);
        newSet.delete(reviewId);
        return newSet;
        });
    }
    };

    const handleReject = async (reviewId, data) => {
    setProcessingItems(prev => new Set([...prev, reviewId]));
    
    try {
        const response = await reviewAPI.submitReview(reviewId, data);
        
        // Remove item from local state
        setReviewItems(prev => prev.filter(item => item.id !== reviewId));
        
        toast.success(`Review rejected successfully! ${response.data.remaining_reviews} reviews remaining.`);
        
        // If no more reviews for this job, show completion message
        if (response.data.remaining_reviews === 0) {
        toast.success(`Job ${response.data.job_id.slice(0, 8)}... completed! All reviews processed.`, {
            duration: 6000,
        });
        }
        
    } catch (error) {
        console.error('Failed to reject review:', error);
        toast.error('Failed to reject review: ' + (error.response?.data?.detail || error.message));
    } finally {
        setProcessingItems(prev => {
        const newSet = new Set(prev);
        newSet.delete(reviewId);
        return newSet;
        });
    }
    };


  const handleRefresh = () => {
    fetchReviewQueue();
    toast.success('Review queue refreshed');
  };

  if (loading) {
    return <PageLoadingSpinner text="Loading review queue..." />;
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Review Queue</h1>
          <p className="text-gray-600 mt-1">
            Review and approve document processing actions
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

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <StatsCard
          title="Total Items"
          value={reviewItems.length}
          subtitle="Pending review"
          icon={ClockIcon}
          color="warning"
        />
        
        <StatsCard
          title="Critical Priority"
          value={getPriorityCount('critical')}
          subtitle="Needs immediate attention"
          icon={ExclamationTriangleIcon}
          color="danger"
        />
        
        <StatsCard
          title="High Priority"
          value={getPriorityCount('high')}
          subtitle="High importance"
          icon={EyeIcon}
          color="warning"
        />
        
        <StatsCard
          title="Processing"
          value={processingItems.size}
          subtitle="Currently being reviewed"
          icon={LoadingSpinner}
          color="primary"
        />
      </div>

      {/* Priority Filter */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">Filter by Priority</h3>
          
          <div className="flex items-center space-x-2">
            {priorityFilters.map((filter) => {
              const count = getPriorityCount(filter.value);
              const isActive = selectedPriority === filter.value;
              
              return (
                <button
                  key={filter.value}
                  onClick={() => setSelectedPriority(filter.value)}
                  className={clsx(
                    'px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
                    isActive
                      ? 'bg-primary-100 text-primary-700 border border-primary-200'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  )}
                >
                  {filter.label} ({count})
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Review Items */}
      <div className="space-y-6">
        {filteredItems.length === 0 ? (
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-12 text-center">
            {reviewItems.length === 0 ? (
              <>
                <CheckCircleIcon className="h-12 w-12 text-green-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">All caught up!</h3>
                <p className="text-gray-600">
                  No items currently need human review. Great job!
                </p>
              </>
            ) : (
              <>
                <EyeIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No items found</h3>
                <p className="text-gray-600">
                  No review items match the selected priority filter.
                </p>
              </>
            )}
          </div>
        ) : (
          <>
            <div className="text-sm text-gray-600">
              Showing {filteredItems.length} of {reviewItems.length} review items
              {selectedPriority !== 'all' && ` with ${selectedPriority} priority`}
            </div>

            {filteredItems.map((item) => (
              <ReviewItem
                key={item.id}
                item={item}
                onApprove={handleApprove}
                onReject={handleReject}
                isProcessing={processingItems.has(item.id)}
              />
            ))}
          </>
        )}
      </div>

      {/* Processing indicator */}
      {processingItems.size > 0 && (
        <div className="fixed bottom-4 right-4 bg-white rounded-lg shadow-lg border border-gray-200 p-3 flex items-center space-x-2">
          <LoadingSpinner size="sm" />
          <span className="text-sm text-gray-600">
            Processing {processingItems.size} review{processingItems.size > 1 ? 's' : ''}...
          </span>
        </div>
      )}

      {/* Auto-refresh indicator */}
      {refreshing && (
        <div className="fixed bottom-4 left-4 bg-white rounded-lg shadow-lg border border-gray-200 p-3 flex items-center space-x-2">
          <LoadingSpinner size="sm" />
          <span className="text-sm text-gray-600">Refreshing queue...</span>
        </div>
      )}
    </div>
  );
}
