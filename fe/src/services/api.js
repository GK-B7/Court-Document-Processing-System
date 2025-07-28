import axios from 'axios';
import toast from 'react-hot-toast';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for adding auth headers (future use)
api.interceptors.request.use(
  (config) => {
    // Add auth token here if needed in the future
    // const token = localStorage.getItem('auth_token');
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`;
    // }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Don't show toast for requests marked as silent
    if (!error.config?.skipErrorToast) {
      const message = error.response?.data?.detail || 
                     error.response?.data?.message || 
                     error.message || 
                     'An error occurred';
      toast.error(message);
    }
    
    // Log error for debugging
    console.error('API Error:', {
      url: error.config?.url,
      method: error.config?.method,
      status: error.response?.status,
      message: error.response?.data?.detail || error.message,
    });
    
    return Promise.reject(error);
  }
);

// ──────────────────────────────────────────────────────────────────────────────
// Job Management APIs
// ──────────────────────────────────────────────────────────────────────────────

export const jobsAPI = {
  /**
   * Upload a document for processing
   * @param {File} file - The file to upload
   * @param {Function} onProgress - Progress callback function
   * @returns {Promise} Upload response with job_id
   */
  uploadDocument: (file, onProgress) => {
    const formData = new FormData();
    formData.append('file', file);
    
    return api.post('/upload', formData, {
      headers: { 
        'Content-Type': 'multipart/form-data' 
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          onProgress(progress);
        }
      },
      timeout: 120000, // 2 minutes for file uploads
    });
  },

  /**
   * Get all jobs with optional filtering
   * @param {Object} params - Query parameters
   * @param {string} params.status - Filter by status (optional)
   * @param {number} params.limit - Number of jobs to return (default: 50)
   * @param {number} params.offset - Pagination offset (default: 0)
   * @returns {Promise} List of jobs
   */
  getAllJobs: (params = {}) => {
    const queryParams = {
      limit: 50,
      offset: 0,
      ...params,
    };
    
    // Remove null/undefined values
    Object.keys(queryParams).forEach(key => {
      if (queryParams[key] === null || queryParams[key] === undefined) {
        delete queryParams[key];
      }
    });
    
    return api.get('/jobs', { params: queryParams });
  },

  /**
   * Get detailed information about a specific job
   * @param {string} jobId - The job ID
   * @returns {Promise} Job details with review items and logs
   */
  getJobById: (jobId) => {
    return api.get(`/jobs/${jobId}`);
  },

  /**
   * Get current status of a job
   * @param {string} jobId - The job ID
   * @returns {Promise} Job status and progress
   */
  getJobStatus: (jobId) => {
    return api.get(`/status/${jobId}`, {
      skipErrorToast: true, // Don't show error toast for polling requests
    });
  },

  /**
   * Get multiple job statuses (for batch updates)
   * @param {string[]} jobIds - Array of job IDs
   * @returns {Promise} Array of job statuses
   */
  batchGetJobStatuses: async (jobIds) => {
    try {
      const promises = jobIds.map(jobId => 
        jobsAPI.getJobStatus(jobId).catch(error => ({ error, jobId }))
      );
      const results = await Promise.all(promises);
      return results;
    } catch (error) {
      console.error('Batch job status fetch failed:', error);
      return [];
    }
  },
};

// ──────────────────────────────────────────────────────────────────────────────
// Review Queue APIs
// ──────────────────────────────────────────────────────────────────────────────

export const reviewAPI = {
  /**
   * Get all items in the review queue
   * @returns {Promise} List of review items
   */
  getReviewQueue: () => {
    return api.get('/review_queue');
  },

  /**
   * Submit a review decision for an item
   * @param {string} reviewId - The review item ID
   * @param {Object} reviewData - Review decision data
   * @param {boolean} reviewData.approved - Whether the item is approved
   * @param {string} reviewData.corrected_id - Corrected national ID (optional)
   * @param {string} reviewData.corrected_action - Corrected action (optional)
   * @param {string} reviewData.comments - Review comments (optional)
   * @returns {Promise} Review submission response
   */
  submitReview: (reviewId, reviewData) => {
    return api.post(`/review/${reviewId}`, reviewData);
  },

  /**
   * Batch submit multiple reviews
   * @param {Array} reviews - Array of review submissions
   * @returns {Promise} Array of submission results
   */
  batchSubmitReviews: async (reviews) => {
    try {
      const promises = reviews.map(review => 
        reviewAPI.submitReview(review.id, review.data)
          .catch(error => ({ error, reviewId: review.id }))
      );
      const results = await Promise.all(promises);
      return results;
    } catch (error) {
      console.error('Batch review submission failed:', error);
      return [];
    }
  },
};

// ──────────────────────────────────────────────────────────────────────────────
// System & Health APIs
// ──────────────────────────────────────────────────────────────────────────────

export const systemAPI = {
  /**
   * Get system health status
   * @returns {Promise} System health information
   */
  getHealth: () => {
    return api.get('/health');
  },

  /**
   * Get system metrics and statistics
   * @returns {Promise} Processing metrics and stats
   */
  getMetrics: () => {
    return api.get('/metrics');
  },

  /**
   * Get all supported actions
   * @returns {Promise} List of available actions
   */
  getActions: () => {
    return api.get('/actions');
  },

  /**
   * Get supported actions for execution
   * @returns {Promise} List of executable actions
   */
  getSupportedActions: () => {
    return api.get('/supported-actions');
  },
};

// ──────────────────────────────────────────────────────────────────────────────
// Customer APIs
// ──────────────────────────────────────────────────────────────────────────────

export const customerAPI = {
  /**
   * Search for customers by national ID or customer ID
   * @param {Object} params - Search parameters
   * @param {string} params.national_id - National ID to search for
   * @param {string} params.customer_id - Customer ID to search for
   * @returns {Promise} Customer information
   */
  searchCustomers: (params) => {
    return api.get('/customers/search', { params });
  },

  /**
   * Search customer by national ID
   * @param {string} nationalId - National ID
   * @returns {Promise} Customer information
   */
  getByNationalId: (nationalId) => {
    return customerAPI.searchCustomers({ national_id: nationalId });
  },

  /**
   * Search customer by customer ID
   * @param {string} customerId - Customer ID
   * @returns {Promise} Customer information
   */
  getByCustomerId: (customerId) => {
    return customerAPI.searchCustomers({ customer_id: customerId });
  },
};

// ──────────────────────────────────────────────────────────────────────────────
// Action Execution APIs
// ──────────────────────────────────────────────────────────────────────────────

export const actionAPI = {
  /**
   * Execute an action for a customer
   * @param {string} customerId - Customer ID
   * @param {string} actionName - Action to execute
   * @returns {Promise} Execution result
   */
  executeAction: (customerId, actionName) => {
    return api.post('/execute-action', null, {
      params: { 
        customer_id: customerId, 
        action_name: actionName 
      },
    });
  },

  /**
   * Execute freeze funds action
   * @param {string} customerId - Customer ID
   * @returns {Promise} Execution result
   */
  freezeFunds: (customerId) => {
    return actionAPI.executeAction(customerId, 'freeze_funds');
  },

  /**
   * Execute release funds action
   * @param {string} customerId - Customer ID
   * @returns {Promise} Execution result
   */
  releaseFunds: (customerId) => {
    return actionAPI.executeAction(customerId, 'release_funds');
  },
};

// ──────────────────────────────────────────────────────────────────────────────
// Utility Functions
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Create a silent request (won't show error toasts)
 * @param {Function} apiCall - The API function to call
 * @param {...any} args - Arguments to pass to the API function
 * @returns {Promise} API response
 */
export const silentRequest = async (apiCall, ...args) => {
  try {
    const response = await apiCall(...args);
    return { data: response.data, error: null };
  } catch (error) {
    return { data: null, error };
  }
};

/**
 * Poll an endpoint until a condition is met
 * @param {Function} apiCall - The API function to poll
 * @param {Function} condition - Condition function that returns true when polling should stop
 * @param {Object} options - Polling options
 * @param {number} options.interval - Polling interval in ms (default: 5000)
 * @param {number} options.maxAttempts - Maximum attempts (default: 60)
 * @returns {Promise} Final result or timeout
 */
export const pollEndpoint = async (apiCall, condition, options = {}) => {
  const { interval = 5000, maxAttempts = 60 } = options;
  let attempts = 0;

  while (attempts < maxAttempts) {
    try {
      const response = await apiCall();
      const result = response.data;
      
      if (condition(result)) {
        return { data: result, attempts };
      }
      
      await new Promise(resolve => setTimeout(resolve, interval));
      attempts++;
    } catch (error) {
      console.error(`Polling attempt ${attempts + 1} failed:`, error);
      attempts++;
      
      if (attempts >= maxAttempts) {
        throw new Error(`Polling failed after ${maxAttempts} attempts`);
      }
      
      await new Promise(resolve => setTimeout(resolve, interval));
    }
  }
  
  throw new Error(`Polling timeout after ${maxAttempts} attempts`);
};

/**
 * Retry an API call with exponential backoff
 * @param {Function} apiCall - The API function to retry
 * @param {Object} options - Retry options
 * @param {number} options.maxRetries - Maximum retry attempts (default: 3)
 * @param {number} options.baseDelay - Base delay in ms (default: 1000)
 * @returns {Promise} API response
 */
export const retryRequest = async (apiCall, options = {}) => {
  const { maxRetries = 3, baseDelay = 1000 } = options;
  let lastError;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await apiCall();
    } catch (error) {
      lastError = error;
      
      if (attempt === maxRetries) {
        break;
      }
      
      const delay = baseDelay * Math.pow(2, attempt);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  
  throw lastError;
};

// ──────────────────────────────────────────────────────────────────────────────
// Export default api instance and all APIs
// ──────────────────────────────────────────────────────────────────────────────

export default api;


// Export commonly used combinations
export const documentProcessingAPI = {
  ...jobsAPI,
  ...reviewAPI,
};

export const administrationAPI = {
  ...systemAPI,
  ...customerAPI,
  ...actionAPI,
};
