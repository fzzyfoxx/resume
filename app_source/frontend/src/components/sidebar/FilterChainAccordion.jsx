// src/components/sidebar/FilterChainAccordion.jsx
import React, { useState, useEffect } from 'react';
import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Typography,
  Box,
  IconButton, // Import Button component
} from '@mui/material';
import Tooltip from '@mui/material/Tooltip';
import DeleteIcon from '@mui/icons-material/Delete';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import HistoryIcon from '@mui/icons-material/History';
import CircularLoader from '../common/CircularLoader';
import AddFilterButton from './AddFilterButton';
import Subtitle from '../common/Subtitle';
import QualificationFilter from '../filters/QualificationFilter';
import LinearProgress from '@mui/material/LinearProgress';
import HideLayer from '../../drawing/HideLayer';

function FilterChainAccordion({ chain, chainIndex, onToggle, renderFilterComponent, mapRef, onRemove}) {
  const [addFilterStatus, setAddFilterStatus] = useState('initial');
  const [qualificationValue, setQualificationValue] = useState(chain.qualificationFilterValue || null);
  const [marker, setMarker] = useState(null);

  // New state to store a snapshot of filter values and qualification value
  const [storedFilterValues, setStoredFilterValues] = useState(null);
  const [storedQualificationValue, setStoredQualificationValue] = useState(null);
  const [hasChanges, setHasChanges] = useState(false); // New state to track if changes have been made

  // Effect to update qualificationValue when parent chain object changes
  useEffect(() => {
    setQualificationValue(chain.qualificationFilterValue || null);
  }, [chain.qualificationFilterValue]);

  // Effect to set initial qualification value and reset if children exist
  useEffect(() => {
    const resetQualificationValueIfChildrenExist = () => {
      if (chain.filters[chain.filters.length - 1]?.children) {
        setQualificationValue(null);
      }
    };
    const SetInitialQualificationValue = () => {
      if (!qualificationValue && !chain.qualificationFilterValue) {
        setQualificationValue({option: 'Excluded', label: 'obszar wykluczony', value: ''});
      }
    };
    SetInitialQualificationValue();
    resetQualificationValueIfChildrenExist();
  }, [chain.filters]);

  // Effect to capture filter values when addFilterStatus turns to 'stop'
  // This establishes the baseline for comparison
  useEffect(() => {
    if (addFilterStatus === 'stop') {
      // Store the current state of filters and qualification
      const currentFilterSelectedValues = chain.filters.map(f => ({
        id: f.id,
        selectedValue: f.selectedValue,
      }));
      setStoredFilterValues(currentFilterSelectedValues);
      setStoredQualificationValue(qualificationValue);
      // setHasChanges(false) is handled by the new effect below
    }
  }, [addFilterStatus, chain.filters, qualificationValue]); // Depend on addFilterStatus and current values

  // Function to compare current values with stored values
  const checkChanges = (currentFilters, currentQualification) => {
    if (!storedFilterValues || !storedQualificationValue) {
      return false; // No stored values to compare against
    }

    // Compare qualification value
    if (JSON.stringify(currentQualification) !== JSON.stringify(storedQualificationValue)) {
      return true;
    }

    // Compare individual filter values
    if (currentFilters.length !== storedFilterValues.length) {
      return true; // Filters added or removed
    }

    for (let i = 0; i < currentFilters.length; i++) {
      const currentFilter = currentFilters[i];
      const storedFilter = storedFilterValues.find(f => f.id === currentFilter.id);

      // If a filter is missing or its selectedValue is different
      if (!storedFilter || JSON.stringify(currentFilter.selectedValue) !== JSON.stringify(storedFilter.selectedValue)) {
        return true;
      }
    }
    return false;
  };

  // NEW EFFECT: To continuously check for changes against the stored state
  useEffect(() => {
    // Only check for changes if a baseline (stored values) exists
    if (storedFilterValues !== null || storedQualificationValue !== null) {
      setHasChanges(checkChanges(chain.filters, qualificationValue));
    }
  }, [chain.filters, qualificationValue, storedFilterValues, storedQualificationValue, checkChanges]); // Depend on all values that can change and the checkChanges function

  const handleFilterValueChange = (filterId, value, symbols, hasChildren, isValueEmpty, filterType) => {
    let newQualificationValue = qualificationValue;
    let updatedFilters = chain.filters;

    if (filterType === 'qualification') {
      newQualificationValue = isValueEmpty ? null : value;
      setQualificationValue(newQualificationValue);
    } else {
      updatedFilters = chain.filters.map((filter) => {
        if (filter.id === filterId) {
          return {
            ...filter,
            selectedValue: isValueEmpty ? null : value,
          };
        }
        return filter;
      });
    }

    const updatedChain = {
      ...chain,
      qualificationFilterValue: newQualificationValue,
      filters: updatedFilters,
    };

    onToggle(updatedChain.id, updatedChain.isExpanded, updatedChain);

    // The setHasChanges is now handled by the new useEffect above,
    // which reacts to changes in chain.filters and qualificationValue.
  };

  const handleRestoreValues = () => {
    if (storedFilterValues && storedQualificationValue) {
      // Revert individual filters
      const restoredFilters = chain.filters.map(filter => {
        const stored = storedFilterValues.find(f => f.id === filter.id);
        return {
          ...filter,
          selectedValue: stored ? stored.selectedValue : null,
        };
      });

      // Revert qualification filter (local state and prop for parent)
      setQualificationValue(storedQualificationValue);

      const restoredChain = {
        ...chain,
        filters: restoredFilters,
        qualificationFilterValue: storedQualificationValue,
      };

      onToggle(restoredChain.id, restoredChain.isExpanded, restoredChain);
      setHasChanges(false); // No changes after restoring
    }
  };

  const accordionTitle = React.useMemo(() => {
    const selectedFilterParts = chain.filters
      .map((f) => {
        if (!f.selectedValue || (Array.isArray(f.selectedValue) && f.selectedValue.length === 0)) {
          return null;
        }
        return f.selector_type === 'combo_box'
          ? f.selectedValue
          : f.title;
      })
      .filter(Boolean);

    if (qualificationValue !== null) {
      selectedFilterParts.push(`${qualificationValue.label}`);
    }

    return selectedFilterParts.length > 0
      ? selectedFilterParts.join(' > ')
      : `NowyFiltr-${chainIndex + 1}`;
  }, [chain.filters, chainIndex, qualificationValue]);

  const renderBottomBar = () => {
    if (addFilterStatus === 'stop') {
      return (
        <Box sx={{ width: 'calc(100% - 14px)', position: 'relative', left: '4px'}}>
                  <LinearProgress
          sx={{
            position: 'relative', // Ensure it can be moved
            left: '4px', // Shift it to the right
            width: '100%-14px', // Adjust the width to fit within the shifted container
          }}
        />
        </Box>
      );
    }

    let backgroundColor = 'transparent';
    if (addFilterStatus === 'update') {
      backgroundColor = 'transparent';
    }

    return (
      <Box
        sx={{
          width: '100%',
          height: '4px',
          backgroundColor
        }}
      />
    );
  };

  const indicatorColor = () => {
    let color = '#e0e0e0'; // lightgray

    if (!hasChanges && addFilterStatus !== 'stop' && addFilterStatus !== 'add' && addFilterStatus !== 'initial') {
      color = '#81c784'; // lightgreen
    } else if (hasChanges) {
      color = '#ffb74d'; // Orange for unsaved changes
    } else if (addFilterStatus === 'stop' && !hasChanges) { // Added !hasChanges to ensure it's lightgreen only if no new changes after stop
      color = '#e0e0e0'; // Lightgreen when stopped and no new changes
    }
    return color;
  };

  const memoizedHasChanges = React.useMemo(() => hasChanges, [hasChanges]);

  return (
    <Box sx={{ position: 'relative' }}>
      <Box
      sx={{
        position: 'absolute',
        top: 0,
        bottom: 0,
        left: 0,
        width: '4px',
        backgroundColor: indicatorColor(),
        zIndex: 1,
        pointerEvents: 'none',
      }}
      />
    <Accordion
      expanded={chain.isExpanded}
      onChange={(event, expanded) => onToggle(chain.id, expanded, { ...chain, isExpanded: expanded })}
      disableGutters
      sx={{
        mt: 0,
        mb: 0,
        boxShadow: 'none',
        borderTop: chainIndex === 0 ? '1px solid #eee' : 'none',
        borderBottom: '1px solid #eee',
        '&.MuiAccordion-root': {
          '&:before': { display: 'none' },
        },
      }}
    >
      <AccordionSummary
        expandIcon={<ExpandMoreIcon />}
        sx={{
          minHeight: '36px !important',
          alignItems: 'center',
          '& .MuiAccordionSummary-content': {
            margin: '0 !important',
            flexGrow: 1,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '0 !important',
          },
          '& .MuiAccordionSummary-root': {
            padding: '0 !important',
          },
          pr: 1,
        }}
      >
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            flexGrow: 1,
            alignItems: 'flex-start',
            ml: 1.5,
            py: 0.5,
          }}
        >
          <Typography
            variant="caption"
            color="textSecondary"
            sx={{
              wordBreak: 'break-word',
            }}
          >
            {accordionTitle}
          </Typography>
          {chain.isLoading && (
            <CircularLoader size={14} sx={{ mt: 0.5, flexShrink: 0 }} />
          )}
        </Box>
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
          }}
        >
          {marker && <HideLayer marker={marker} mapRef={mapRef} />}
        </Box>
      </AccordionSummary>
      <AccordionDetails sx={{ p: 0 }}>
        {chain.filters.length > 0 && (
          <Box
            sx={{
              width: '90%',
              maxWidth: 'calc(100% - 24px)',
              display: 'flex',
              justifyContent: 'left',
              my: 0.5,
              ml: 2.0,
              px: 1,
              paddingLeft: '14px',
              paddingRight: '12px',
              marginBottom: '4px',
              marginTop: '14px',
            }}
          >
            <Subtitle label="Objects definition" />
          </Box>
        )}
        {chain.filters.map((filter) => {
          // Calculate disabled state for each filter individually
          
          return (
            <Box key={filter.id} sx={{ mb: 0.5 }}>
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: 'left',
                  my: 0.5,
                  ml: 2.0,
                  paddingLeft: '14px',
                }}
              >
                <Box sx={{ width: '90%', maxWidth: 'calc(100% - 24px)' }}>
                  {renderFilterComponent(
                    chain.id, 
                    { ...filter, type: filter.type , addFilterStatus},
                  )}
                </Box>
              </Box>
            </Box>
          );
        })}
        {chain.filters.length > 0 && !chain.filters[chain.filters.length - 1].children && (
          <>
            <Box
              sx={{
                width: '90%',
                maxWidth: 'calc(100% - 24px)',
                display: 'flex',
                justifyContent: 'left',
                my: 0.5,
                ml: 2.0,
                px: 1,
                paddingLeft: '14px',
                paddingRight: '12px',
                marginBottom: '4px',
                marginTop: '24px',
              }}
            >
              <Subtitle label="Qualification" />
            </Box>
            <Box
              sx={{
                width: '90%',
                maxWidth: 'calc(100% - 24px)',
                display: 'flex',
                justifyContent: 'left',
                my: 0.5,
                ml: 2.0,
                px: 1,
                paddingLeft: '14px',
                paddingRight: '12px',
              }}
            >
              <QualificationFilter
                filterId={`${chain.id}-qualification`}
                defaultValue={qualificationValue}
                compact={true}
                onValueChange={(filterId, value, isValueEmpty) =>
                  handleFilterValueChange(filterId, value, null, null, isValueEmpty, 'qualification')
                }
              />
            </Box>
            <Box
                sx={{
                  width: '90%',
                  maxWidth: 'calc(100% - 53px)',
                  display: 'flex',
                  justifyContent: 'right',
                  gap: 2, // Add spacing between buttons
                  my: 1,
                  ml: 2.0,
                  mr: 2.0
                }}
              >
              <Tooltip title="usuń filtr">
              <IconButton
                  onClick={() => {
                    onRemove(chain.id, marker); // Call the onRemove function after removing the layer
                  }}
                  sx={{
                    backgroundColor: 'gray', // Red background
                    '&:hover': {
                      backgroundColor: 'darkgray', // Darker red on hover
                    },
                    color: 'white',
                    borderRadius: '50%',
                    width: 32,
                    height: 32,
                  }}
                >
                  <DeleteIcon />
                </IconButton>
                </Tooltip>
                {hasChanges && (
                <Tooltip title="cofnij zmiany">
                <IconButton
                  onClick={handleRestoreValues}
                  sx={{
                    backgroundColor: 'gray', // Red background
                    '&:hover': {
                      backgroundColor: 'darkgray', // Darker red on hover
                    },
                    color: 'white',
                    borderRadius: '50%',
                    width: 32,
                    height: 32,
                  }}
                >
                  <HistoryIcon />
                </IconButton>
                </Tooltip>
                )}
                
                  <AddFilterButton
                    filters={chain.filters}
                    qualification={qualificationValue}
                    onStatusChange={setAddFilterStatus}
                    mapRef={mapRef}
                    accordionSummary={accordionTitle}
                    marker={marker}
                    setMarker={setMarker}
                    hasChanges={memoizedHasChanges} // Pass hasChanges as a prop
                  />
              </Box>
          </>
        )}
      </AccordionDetails>
      {renderBottomBar()}
    </Accordion>
    </Box>
  );
}

export default FilterChainAccordion;

