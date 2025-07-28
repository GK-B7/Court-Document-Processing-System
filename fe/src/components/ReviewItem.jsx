import React, { useState } from 'react';
import {
  EyeIcon,
  CheckCircleIcon,
  XCircleIcon,
  ExclamationTriangleIcon,
  UserIcon,
  DocumentTextIcon,
  ClockIcon,
} from '@heroicons/react/24/outline';
import clsx from 'clsx';
import { formatDistanceToNow } from 'date-fns';

const priorityConfig = {
  critical: { color: 'red', label: 'Critical', bgColor: 'bg-red-100', textColor: 'text-red-800' },
  high: { color: 'orange', label: 'High', bgColor: 'bg-orange-100', textColor: 'text-orange-800' },
  medium: { color: 'yellow', label: 'Medium', bgColor: 'bg-yellow-100', textColor: 'text-yellow-800' },
  low: { color: 'green', label: 'Low', bgColor: 'bg-green-100', textColor: 'text-green-800' },
};

export default function ReviewItem({ 
  item, 
  onApprove, 
  onReject, 
  isProcessing = false,
  className = '' 
}) {
  const [showDetails, setShowDetails] = useState(false);
  const [reviewNotes, setReviewNotes] = useState('');
  const [correctedId, setCorrectedId] = useState(item.national_id || '');
  const [correctedAction, setCorrectedAction] = useState(item.matched_action || '');

  const priority = priorityConfig[item.priority] || priorityConfig.medium;

  const handleApprove = () => {
    onApprove(item.id, {
      approved: true,
      corrected_id: correctedId !== item.national_id ? correctedId : null,
      corrected_action: correctedAction !== item.matched_action ? correctedAction : null,
      comments: reviewNotes,
    });
  };

  const handleReject = () => {
    onReject(item.id, {
      approved: false,
      comments: reviewNotes || 'Rejected by reviewer',
    });
  };

  const formatDate = (dateString) => {
    try {
      return formatDistanceToNow(new Date(dateString), { addSuffix: true });
    } catch {
      return 'Unknown';
    }
  };

  return (
    <div className={clsx(
      'bg-white rounded-xl border border-gray-200 shadow-sm hover:shadow-md transition-all duration-200',
      className
    )}>
      {/* Header */}
      <div className="p-6 border-b border-gray-100">
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-warning-100 rounded-lg">
              <EyeIcon className="h-5 w-5 text-warning-600" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900">
                Review Required
              </h3>
              <div className="flex items-center space-x-2 mt-1">
                <span className={clsx(
                  'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium',
                  priority.bgColor,
                  priority.textColor
                )}>
                  {priority.label} Priority
                </span>
                <span className="text-sm text-gray-500">
                  {formatDate(item.created_at)}
                </span>
              </div>
            </div>
          </div>

          <button
            onClick={() => setShowDetails(!showDetails)}
            className="text-primary-600 hover:text-primary-700 text-sm font-medium"
          >
            {showDetails ? 'Hide Details' : 'Show Details'}
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="p-6">
        {/* Customer & Action Info */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div className="space-y-3">
            <div className="flex items-center space-x-2">
              <UserIcon className="h-4 w-4 text-gray-500" />
              <span className="text-sm font-medium text-gray-700">Customer Info</span>
            </div>
            <div className="bg-gray-50 rounded-lg p-3 space-y-2">
              <div>
                <p className="text-xs text-gray-600">National ID</p>
                <p className="font-mono text-sm text-gray-900">{item.national_id}</p>
              </div>
              {item.customer_id && (
                <div>
                  <p className="text-xs text-gray-600">Customer ID</p>
                  <p className="font-mono text-sm text-gray-900">{item.customer_id}</p>
                </div>
              )}
              {item.customer_name && (
                <div>
                  <p className="text-xs text-gray-600">Customer Name</p>
                  <p className="text-sm text-gray-900">{item.customer_name}</p>
                </div>
              )}
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex items-center space-x-2">
              <DocumentTextIcon className="h-4 w-4 text-gray-500" />
              <span className="text-sm font-medium text-gray-700">Action Details</span>
            </div>
            <div className="bg-gray-50 rounded-lg p-3 space-y-2">
              <div>
                <p className="text-xs text-gray-600">Original Action</p>
                <p className="text-sm text-gray-900">{item.original_action}</p>
              </div>
              <div>
                <p className="text-xs text-gray-600">Matched Action</p>
                <p className="text-sm text-gray-900 font-medium">{item.matched_action}</p>
              </div>
              <div>
                <p className="text-xs text-gray-600">Confidence</p>
                <div className="flex items-center space-x-2">
                  <div className="flex-1 bg-gray-200 rounded-full h-1.5">
                    <div 
                      className="bg-primary-500 h-1.5 rounded-full" 
                      style={{ width: `${(item.confidence || 0) * 100}%` }}
                    />
                  </div>
                  <span className="text-sm text-gray-900">
                    {Math.round((item.confidence || 0) * 100)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Context */}
        {item.context && (
          <div className="mb-6">
            <h4 className="text-sm font-medium text-gray-700 mb-2">Document Context</h4>
            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-sm text-gray-900 italic">"{item.context}"</p>
              {item.page_number && (
                <p className="text-xs text-gray-600 mt-1">Page {item.page_number}</p>
              )}
            </div>
          </div>
        )}

        {/* Review Reasons */}
        {item.review_reasons && item.review_reasons.length > 0 && (
          <div className="mb-6">
            <h4 className="text-sm font-medium text-gray-700 mb-2">Review Reasons</h4>
            <div className="flex flex-wrap gap-2">
              {item.review_reasons.map((reason, index) => (
                <span 
                  key={index}
                  className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-warning-100 text-warning-800"
                >
                  {reason.replace('_', ' ').toLowerCase().replace(/\b\w/g, l => l.toUpperCase())}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Detailed Information */}
        {showDetails && (
          <div className="mb-6 p-4 bg-gray-50 rounded-lg">
            <h4 className="text-sm font-medium text-gray-700 mb-3">Additional Details</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div>
                <p className="text-gray-600">Job ID</p>
                <p className="font-mono text-gray-900">{item.job_id.slice(0, 8)}...</p>
              </div>
              <div>
                <p className="text-gray-600">Risk Level</p>
                <p className="text-gray-900 capitalize">{item.risk_level}</p>
              </div>
              <div>
                <p className="text-gray-600">Similarity Score</p>
                <p className="text-gray-900">{Math.round((item.similarity_score || 0) * 100)}%</p>
              </div>
            </div>
            {item.risk_factors && item.risk_factors.length > 0 && (
              <div className="mt-3">
                <p className="text-gray-600 text-sm mb-1">Risk Factors</p>
                <ul className="text-sm text-gray-900 space-y-1">
                  {item.risk_factors.map((factor, index) => (
                    <li key={index} className="flex items-center">
                      <ExclamationTriangleIcon className="h-3 w-3 text-warning-500 mr-1" />
                      {factor}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Review Form */}
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-gray-700">Review & Corrections</h4>
          
          {/* Correction Fields */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Corrected National ID
              </label>
              <input
                type="text"
                value={correctedId}
                onChange={(e) => setCorrectedId(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
                placeholder="Enter correct National ID"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Corrected Action
              </label>
              <select
                value={correctedAction}
                onChange={(e) => setCorrectedAction(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
              >
                <option value="freeze_funds">Freeze Funds</option>
                <option value="release_funds">Release Funds</option>
              </select>
            </div>
          </div>

          {/* Review Notes */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Review Notes
            </label>
            <textarea
              value={reviewNotes}
              onChange={(e) => setReviewNotes(e.target.value)}
              rows={3}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
              placeholder="Add your review comments..."
            />
          </div>

          {/* Action Buttons */}
          <div className="flex items-center justify-end space-x-3 pt-4 border-t border-gray-200">
            <button
              onClick={handleReject}
              disabled={isProcessing}
              className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-lg text-gray-700 bg-white hover:bg-gray-50 focus:ring-2 focus:ring-red-500 focus:border-red-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <XCircleIcon className="h-4 w-4 mr-2" />
              Reject
            </button>
            <button
              onClick={handleApprove}
              disabled={isProcessing}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-lg text-white bg-green-600 hover:bg-green-700 focus:ring-2 focus:ring-green-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <CheckCircleIcon className="h-4 w-4 mr-2" />
              {isProcessing ? 'Processing...' : 'Approve'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
