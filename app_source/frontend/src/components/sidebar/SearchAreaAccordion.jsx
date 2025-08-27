import React, { useState, useMemo, useEffect, useCallback, useRef } from 'react';
import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Typography,
  Box,
  LinearProgress,
  IconButton,
  Tooltip,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import HistoryIcon from '@mui/icons-material/History';
import AddFilterButton from './AddFilterButton';
import SearchFilterForDicts from '../filters/SearchFilterForDicts';

function SearchAreaAccordion({ mapRef }) {
  const [expanded, setExpanded] = useState(true);
  const [selectedValue, setSelectedValue] = useState(null);

  const [addFilterStatus, setAddFilterStatus] = useState('initial');
  const [marker, setMarker] = useState(null);
  const [storedFilterValues, setStoredFilterValues] = useState(null);
  const [hasChanges, setHasChanges] = useState(false);
  const [implied, setImplied] = useState(false);

  useEffect(() => {
      //console.log(`SearchAreaAccordion: implied changed to ${implied} | addFilterStatus: ${addFilterStatus}`);
    if (addFilterStatus === 'update' && implied) {
      // Store the current state of filters and qualification
      const currentFilterSelectedValues = selectedValue;
      setStoredFilterValues(currentFilterSelectedValues);
    }
  }, [addFilterStatus, implied]);

  const checkChanges = useCallback(
    (currentFilters) => {
      if (!storedFilterValues) {
        return false; // No stored values to compare against
      }

      if (addFilterStatus === 'update') {
        if (JSON.stringify(currentFilters) !== JSON.stringify(storedFilterValues)) {
          return true;
        }
        return false;
      }

      return false;
    },
    [storedFilterValues, addFilterStatus]
  );

  useEffect(() => {
    // Only check for changes if a baseline (stored values) exists
    if (storedFilterValues !== null) {
      setHasChanges(checkChanges(selectedValue));
    }
  }, [selectedValue, storedFilterValues, checkChanges, addFilterStatus]);

  const memoizedHasChanges = useMemo(() => hasChanges, [hasChanges]);

  const handleRestoreValues = () => {
    if (storedFilterValues) {
      setSelectedValue(storedFilterValues);
      setHasChanges(false); // No changes after restoring
    }
  };

  const renderBottomBar = () => {
    if (addFilterStatus === 'stop') {
      return (
        <Box sx={{ width: 'calc(100% - 14px)', position: 'relative', left: '4px' }}>
          <LinearProgress sx={{ position: 'relative', left: '4px', width: '100%-14px' }} />
        </Box>
      );
    }
    return null;
  };

  const handleToggle = () => {
    setExpanded((prev) => !prev);
  };

  const filterId = 'static_search_area_filter';
  const filterTitle = 'jednostki administracyjne';
  const filterSymbols = ['area', 'teryts', 'demo', 'teryts'];

  const addFilterButtonProps = useMemo(() => {
    return {
      filters: [
        {
          id: filterId,
          selector_type: 'search',
          symbolsForNextCall: filterSymbols,
          selectedValue: selectedValue,
        },
      ],
      qualification: { option: 'SearchArea', value: null },
    };
  }, [selectedValue]);

  const handleFilterValueChange = (id, value) => {
    setSelectedValue(value);
  };

  return (
    <Accordion
      expanded={expanded}
      onChange={handleToggle}
      disableGutters
      sx={{
        boxShadow: 'none',
        '&.MuiAccordion-root': { border: 'none', '&:before': { display: 'none' } },
      }}
    >
      <AccordionSummary
        expandIcon={<ExpandMoreIcon />}
        sx={{
          minHeight: '40px !important',
          height: '40px !important',
          '& .MuiAccordionSummary-content': { m: '0 !important', flexGrow: 1 },
          '& .MuiAccordionSummary-root': { p: '0 !important' },
          pr: 1,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', ml: 1.5, py: 0.75 }}>
          <Typography variant="h8">Obszar wyszukiwania</Typography>
        </Box>
      </AccordionSummary>
      <AccordionDetails sx={{ p: 0 }}>
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
          <SearchFilterForDicts
            filterId={filterId}
            title={filterTitle}
            symbols={filterSymbols}
            defaultValue={selectedValue}
            onValueChange={handleFilterValueChange}
            isLoading={false}
            compact={true}
            keysToInclude={['name', 'unit_name', 'teryt']}
          />
        </Box>
        <Box
          sx={{
            width: '90%',
            maxWidth: 'calc(100% - 53px)',
            display: 'flex',
            justifyContent: 'right',
            gap: 2,
            my: 1,
            ml: 2.0,
            mr: 2.0,
          }}
        >
          {hasChanges && (
            <Tooltip title="Cofnij zmiany">
              <IconButton
                onClick={handleRestoreValues}
                sx={{
                  backgroundColor: 'gray',
                  '&:hover': { backgroundColor: 'darkgray' },
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
            filters={addFilterButtonProps.filters}
            qualification={addFilterButtonProps.qualification}
            onStatusChange={setAddFilterStatus}
            onImpliedChange={setImplied}
            mapRef={mapRef}
            accordionSummary="Obszar wyszukiwania"
            marker={marker}
            setMarker={setMarker}
            hasChanges={memoizedHasChanges}
            endpoint="set_search_area"
          />
        </Box>
      </AccordionDetails>
      {renderBottomBar()}
    </Accordion>
    
  );
}

export default SearchAreaAccordion;

