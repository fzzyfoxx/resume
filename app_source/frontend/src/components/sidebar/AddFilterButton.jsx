import React, { useState, useEffect } from 'react';
import Tooltip from '@mui/material/Tooltip';
import IconButton from '@mui/material/IconButton';
import AddIcon from '@mui/icons-material/Add';
import StopIcon from '@mui/icons-material/Stop';
import RefreshIcon from '@mui/icons-material/Refresh';
import axios from 'axios';
import { addShapesFromQuery } from '../../drawing/addShapesFromQuery'; // Import the addShapesFromQuery function

const AddFilterButton = ({ filters, qualification, onStatusChange, mapRef, accordionSummary, marker, setMarker, hasChanges}) => {
  const [status, setStatus] = useState('add'); // Possible values: 'add', 'stop', 'update'
  const [queryId, setQueryId] = useState(null);
  const [pollingInterval, setPollingInterval] = useState(null);
  const [previousStatus, setPreviousStatus] = useState('add'); // Track the previous status
  //const [marker, setMarker] = useState(null); // Marker for rendering shapes
  const [hasUpdated, setHasUpdated] = useState(false);
  console.log('AddFilterButton - hasChanges:', hasChanges);

  useEffect(() => {
    if (onStatusChange) {
      onStatusChange(status);
    }
  }, [status, onStatusChange]);

  useEffect(() => {
    if (status === 'update' && !hasUpdated) {
      const callAddShapes = async () => {
        try {
          const updatedMarker = await addShapesFromQuery(mapRef, {
            query_id: queryId,
            qualification,
            name: accordionSummary,
          }, marker);
          setMarker(updatedMarker); // Update the marker with the new shapes
          setHasUpdated(true); // Mark as updated after calling addShapesFromQuery
        } catch (error) {
          console.error('Error adding shapes from query:', error);
        }
      };
  
      callAddShapes();
    }
  }, [status, hasUpdated, queryId, mapRef, qualification, accordionSummary]);
  
  useEffect(() => {
    if (status !== 'update') {
      setHasUpdated(false); // Reset the flag when status changes away from "update"
    }
  }, [status]);

  const handleAddOrUpdate = async () => {
    try {
      // Prepare the payload
      const payload = {
        filters: filters
          .filter((filter) => !filter.children)
          .map((filter) => ({
            filterId: filter.id,
            selector_type: filter.selector_type,
            symbols: filter.symbolsForNextCall || [],
            values: filter.selectedValue || {},
          })),
        qualification: qualification
          ? {
              option: qualification.option,
              value: qualification.value,
            }
          : null,
      };

      // Call the /calculate_filters endpoint
      const response = await axios.post('http://127.0.0.1:5000/api/queries/calculate_filters', payload, { withCredentials: true });

      if (response.data && response.data.query_id) {
        setPreviousStatus(status); // Save the current status before transitioning to 'stop'
        setQueryId(response.data.query_id);
        setStatus('stop'); // Change status to 'stop' to start polling
      }
    } catch (error) {
      console.error('Error calculating filters:', error);
    }
  };

  const handleStop = () => {
    clearPolling();
    setStatus(previousStatus); // Restore the previous status
    setQueryId(null); // Clear the queryId
  };

  const checkQueryStatus = async () => {
    if (!queryId) return;

    try {
      // Call the /check_query_status endpoint
      const response = await axios.get('http://127.0.0.1:5000/api/queries/check_query_status', {
        params: { query_id: queryId }, withCredentials: true
      });

      if (response.data && response.data.status === 'completed') {
        setStatus('update'); // Change status to 'update' when query is completed
        clearPolling(); // Stop polling once the query is completed
      }
    } catch (error) {
      console.error('Error checking query status:', error);
    }
  };

  const clearPolling = () => {
    if (pollingInterval) {
      clearInterval(pollingInterval);
      setPollingInterval(null);
    }
  };

  const startPolling = (interval = 2000) => {
    const timeoutId = setTimeout(async () => {
      await checkQueryStatus();
      const nextInterval = Math.min(interval + 1000, 10000); // Increase interval up to 10 seconds
      startPolling(nextInterval); // Recursively call startPolling with the updated interval
    }, interval);
    setPollingInterval(timeoutId);
  };

  useEffect(() => {
    if (status === 'stop' && !pollingInterval) {
      startPolling(); // Start polling with the initial interval
    }

    if (status !== 'stop' && pollingInterval) {
      clearPolling(); // Stop polling if status changes from 'stop'
    }

    return () => {
      clearPolling(); // Cleanup on component unmount or status change
    };
  }, [status, queryId]);

  const getIcon = () => {
    switch (status) {
      case 'add':
        return <AddIcon sx={{ color: 'white' }} />;
      case 'stop':
        return <StopIcon sx={{ color: 'white' }} />;
      case 'update':
        return <RefreshIcon sx={{ color: 'white' }} />;
      default:
        return <AddIcon sx={{ color: 'white' }} />;
    }
  };

  return (
    <Tooltip title={status === 'add' ? "dodaj filtr" 
        : status === 'update' && !hasChanges ? "brak zmian do odświeżenia"
        : status === 'update' ? "odśwież filtr" 
        : status === 'stop' ? "zatrzymaj" : ""}
        >
        <span>
    <IconButton
      onClick={
        status === 'update' && !hasChanges ? null
        : status === 'add' || status === 'update' ? handleAddOrUpdate
        : status === 'stop' ? handleStop
        : null
      }
      sx={{
        backgroundColor: status === 'update' && !hasChanges ? 'lightgray' : 'gray', // Change color when disabled lightgray
        '&:hover': {
          backgroundColor: status === 'update' && !hasChanges ? 'lightgray' : 'darkgray', // Prevent hover effect when disabled
        },
        color: 'white',
        borderRadius: '50%',
        width: 32,
        height: 32,
      }}
    >
      {getIcon()}
    </IconButton>
    </span>
    </Tooltip>
  );
};

export default AddFilterButton;