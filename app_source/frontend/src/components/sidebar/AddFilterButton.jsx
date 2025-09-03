import React, { useState, useEffect, useMemo } from 'react';
import Tooltip from '@mui/material/Tooltip';
import IconButton from '@mui/material/IconButton';
import AddIcon from '@mui/icons-material/Add';
import StopIcon from '@mui/icons-material/Stop';
import RefreshIcon from '@mui/icons-material/Refresh';
import axios from 'axios';
import { addShapesFromQuery } from '../../drawing/addShapesFromQuery'; // Import the addShapesFromQuery function
import { generateUniqueId } from '../../utils/idGenerator';
import { getDynamicButtonStyle } from '../../styles/ButtonStyles';
import { useFilterQuery } from '../../hooks/useFilterQuery';

const AddFilterButton = ({ 
  filters, 
  status, 
  hasChanges, 
  isMain, 
  isActual,
  stateId, 
  handleAddOrUpdate,
  handleStop,
  }) => {

  const iconSize = 20;

  const getIcon = () => {
    switch (status) {
      case 'add':
        return <AddIcon sx={{ fontSize: iconSize }} />;
      case 'stop':
        return <StopIcon sx={{ fontSize: iconSize }} />;
      case 'update':
        return <RefreshIcon sx={{ fontSize: iconSize }} />;
      default:
        return <AddIcon sx={{ fontSize: iconSize }} />;
    }
  };

  const isEmpty = (filters) => {
    return !filters.some(f => f.selectedValue && Object.keys(f.selectedValue).length > 0);
  };
  
  const isFiltersEmpty = useMemo(() => isEmpty(filters), [filters]);

  const isDisabled = useMemo(() => 
    (status === 'update' && !hasChanges && isActual) || isFiltersEmpty || (!stateId && !isMain),
    [status, hasChanges, isActual, isFiltersEmpty, stateId, isMain]
  );


  return (
    <Tooltip title={isFiltersEmpty ? "ustaw wartości"
        : (!stateId && !isMain) ? "wybierz obszar wyszukiwania"
        : !isActual ? "odśwież dla nowego obszaru"
        : status === 'add' ? "dodaj" 
        : status === 'update' && !hasChanges ? "brak zmian do odświeżenia"
        : status === 'update' ? "odśwież" 
        : status === 'stop' ? "zatrzymaj" : ""}
        >
        <span>
    <IconButton
      onClick={
         (!stateId && !isMain) ? null
         : status === 'stop' ? handleStop
        : isFiltersEmpty ? null
        : !isActual ? handleAddOrUpdate
        : status === 'update' && !hasChanges ? null
        : status === 'add' || status === 'update' ? handleAddOrUpdate
        : null
      }
      sx={getDynamicButtonStyle({disabled: isDisabled, isMainButton: true})}
    >
      {getIcon()}
    </IconButton>
    </span>
    </Tooltip>
  );
};

export default AddFilterButton;