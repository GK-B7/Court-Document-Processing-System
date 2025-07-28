import React from 'react';
import { 
  ClockIcon, 
  CheckCircleIcon, 
  ExclamationTriangleIcon, 
  XCircleIcon,
  CogIcon,
  BoltIcon,
  ShieldCheckIcon,
  BanknotesIcon,
  LockClosedIcon,
  LockOpenIcon
} from '@heroicons/react/24/outline';
import { formatRelativeTime, getStatusColor } from '../utils/helpers';
import clsx from 'clsx';

export default function JobStatus({ 
  job, 
  showDetails = true, 
  showProgress = false, 
  className = '', 
  onClick 
}) {
  const getStatusIcon = (status) => {
    const icons = {
      pending: ClockIcon,
      processing: CogIcon,
      completed: CheckCircleIcon,
      review_required: ExclamationTriangleIcon,
      failed: XCircleIcon,
    };
    return icons[status] || ClockIcon;
  };

  const StatusIcon = getStatusIcon(job.status);
  const statusColor = getStatusColor(job.status);

  // Get action icon based on action name
  const getActionIcon = (actionName) => {
    const icons = {
      'freeze_funds': LockClosedIcon,
      'release_funds': LockOpenIcon,
      'close_account': XCircleIcon,
      'open_account': CheckCircleIcon,
      'default': BanknotesIcon
    };
    return icons[actionName] || icons['default'];
  };

  // Enhanced action extraction from job metadata
  const getJobActions = (job) => {
    const metadata = job.metadata || {};
    const actionsSummary = metadata.actions_summary || {};
    const actions = [];

    // 1. Executed actions (auto-approved or from reviews)
    const executedActions = actionsSummary.executed_actions || [];
    executedActions.forEach(action => {
      actions.push({
        type: 'executed',
        name: action.action_name,
        customer_id: action.customer_id,
        national_id: action.national_id,
        status: action.status,
        executed_at: action.executed_at,
        icon: getActionIcon(action.action_name)
      });
    });

    // 2. Reviewed and executed actions
    const reviewedActions = actionsSummary.reviewed_actions || [];
    reviewedActions.forEach(action => {
      actions.push({
        type: 'reviewed_executed',
        name: action.action_name,
        customer_id: action.customer_id,
        national_id: action.national_id,
        status: action.status,
        executed_at: action.executed_at,
        icon: getActionIcon(action.action_name)
      });
    });

    // 3. Failed actions
    const failedActions = actionsSummary.failed_actions || [];
    failedActions.forEach(action => {
      actions.push({
        type: 'failed',
        name: action.action_name,
        customer_id: action.customer_id,
        national_id: action.national_id,
        status: 'failed',
        error: action.error,
        icon: getActionIcon(action.action_name)
      });
    });

    // 4. Pending review actions
    const pendingReviews = actionsSummary.pending_reviews || [];
    pendingReviews.forEach(action => {
      actions.push({
        type: 'pending_review',
        name: action.action_name,
        customer_id: action.customer_id,
        national_id: action.national_id,
        status: 'pending_review',
        confidence: action.confidence,
        icon: getActionIcon(action.action_name)
      });
    });

    return actions;
  };

  const actions = getJobActions(job);

  return (
    <div 
      className={clsx(
        'bg-white border border-gray-200 rounded-xl p-4 shadow-sm hover:shadow-md transition-all duration-200',
        onClick && 'cursor-pointer hover:border-primary-300',
        className
      )}
      onClick={onClick}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-start space-x-3 flex-1">
          <div className={clsx(
            'p-2 rounded-lg',
            statusColor === 'green' && 'bg-green-100 text-green-600',
            statusColor === 'blue' && 'bg-blue-100 text-blue-600',
            statusColor === 'yellow' && 'bg-yellow-100 text-yellow-600',
            statusColor === 'red' && 'bg-red-100 text-red-600',
            statusColor === 'gray' && 'bg-gray-100 text-gray-600'
          )}>
            <StatusIcon className="h-5 w-5" />
          </div>
          
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between mb-1">
              <h3 className="text-sm font-medium text-gray-900 truncate">
                {job.filename || `Job ${job.job_id.slice(0, 8)}...`}
              </h3>
              <span className={clsx(
                'inline-flex items-center px-2 py-1 rounded-full text-xs font-medium',
                statusColor === 'green' && 'bg-green-100 text-green-800',
                statusColor === 'blue' && 'bg-blue-100 text-blue-800',
                statusColor === 'yellow' && 'bg-yellow-100 text-yellow-800',
                statusColor === 'red' && 'bg-red-100 text-red-800',
                statusColor === 'gray' && 'bg-gray-100 text-gray-800'
              )}>
                {job.status.replace('_', ' ')}
              </span>
            </div>

            {/* SPECIFIC ACTIONS DISPLAY */}
            {job.status === 'failed' ? (
              <div className="mb-2">
                <div className="flex items-center space-x-2">
                  <XCircleIcon className="h-4 w-4 text-red-500" />
                  <span className="text-xs text-red-600 truncate" title={job.error_message}>
                    Error: {job.error_message || 'Processing failed'}
                  </span>
                </div>
              </div>
            ) : actions.length > 0 ? (
              <div className="mb-2">
                <div className="flex items-center space-x-2 mb-2">
                  <BoltIcon className="h-4 w-4 text-indigo-500" />
                  <span className="text-xs font-medium text-indigo-700">Banking Actions:</span>
                </div>
                <div className="space-y-1">
                  {actions.map((action, index) => {
                    const ActionIcon = action.icon;
                    return (
                      <div key={index} className={clsx(
                        'inline-flex items-center px-2 py-1 rounded-md text-xs font-medium mr-2 mb-1',
                        action.type === 'executed' && 'bg-green-100 text-green-800 border border-green-200',
                        action.type === 'reviewed_executed' && 'bg-blue-100 text-blue-800 border border-blue-200',
                        action.type === 'failed' && 'bg-red-100 text-red-800 border border-red-200',
                        action.type === 'pending_review' && 'bg-yellow-100 text-yellow-800 border border-yellow-200'
                      )}>
                        <ActionIcon className="h-3 w-3 mr-1" />
                        <span className="font-semibold capitalize">
                          {action.name.replace('_', ' ')}
                        </span>
                        {action.status === 'failed' && (
                          <XCircleIcon className="h-3 w-3 ml-1 text-red-500" />
                        )}
                        {action.status === 'success' && (
                          <CheckCircleIcon className="h-3 w-3 ml-1 text-green-500" />
                        )}
                        {action.customer_id && (
                          <span className="ml-1 text-xs opacity-75">
                            ({action.customer_id.toString().slice(-4)})
                          </span>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="mb-2">
                <div className="flex items-center space-x-2">
                  <BoltIcon className="h-4 w-4 text-gray-400" />
                  <span className="text-xs text-gray-500">No banking actions executed</span>
                </div>
              </div>
            )}
            
            <div className="flex items-center text-xs text-gray-500 space-x-4">
              <span>
                Created {formatRelativeTime(job.created_at)}
              </span>
              
              {showDetails && job.metadata?.actions_summary && (
                <>
                  <span>{job.metadata.actions_summary.total_actions_found || 0} actions found</span>
                  <span>{job.metadata.actions_summary.total_executed || 0} executed</span>
                </>
              )}
            </div>

            {showProgress && job.progress !== undefined && (
              <div className="mt-2">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-xs text-gray-600">Progress</span>
                  <span className="text-xs text-gray-600">
                    {Math.round((job.progress || 0) * 100)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-1.5">
                  <div 
                    className={clsx(
                      'h-1.5 rounded-full transition-all duration-300',
                      statusColor === 'green' && 'bg-green-500',
                      statusColor === 'blue' && 'bg-blue-500',
                      statusColor === 'yellow' && 'bg-yellow-500',
                      statusColor === 'red' && 'bg-red-500',
                      statusColor === 'gray' && 'bg-gray-500'
                    )}
                    style={{ width: `${Math.round((job.progress || 0) * 100)}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
