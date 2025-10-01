import React, {useState, useMemo, useEffect, useRef, useCallback} from 'react';
import { Accordion, AccordionSummary, AccordionDetails, Box, Typography, IconButton, Tooltip } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { useFilterChainState } from '../../hooks/checkChanges';
import { generateAccordionTitle, getIndicatorColor, getAccordionSummaryParts } from '../../utils/filterChainUtils';
import { renderBottomBar } from '../common/horizontalLoader';
import FilterChainButtons from './bottomButtons';
import AccordionSummaryContent from './AccordionSummaryContent';
import { generateUniqueId } from '../../utils/idGenerator';
import { addShapesFromQuery } from '../../drawing/addShapesFromQuery'; // Import the function
import AccordionStatusButton from './AccordionStatusButton';
import { useFilterQuery } from '../../hooks/useFilterQuery';
import { useDebounce } from '../../hooks/useDebounce';

function FilterChainAccordion2({ 
    chain, 
    chainIndex, 
    onToggle, 
    renderFilterComponent, 
    mapRef, 
    onRemove, 
    calculation_endpoint, 
    showDeleteButton = true,
    labelVariant = "caption",
    accordionLabel = null,
    staticLabel = false,
    isMain = false,
    stateId,
    setStateId,
    onStateChange,
    fitBounds,
    loadedChainName,
    onMarkerCreated,
    allFilterStateIds,
    sourceStateId,
    setSourceStateId,
    isTarget = false,
    onBulkUpdate,
    onBulkStop,
    isLayerVisible,
    setLayerVisible
    }) {
  const [addFilterStatus, setAddFilterStatus] = useState('add');
  const debouncedAddFilterStatus = useDebounce(addFilterStatus, 300);
  const [lastProcessedBulkUpdate, setLastProcessedBulkUpdate] = useState(null);
  const [lastProcessedBulkStop, setLastProcessedBulkStop] = useState(null);

    // LOG: Track status changes within the child accordion
    useEffect(() => {
      console.log(`[FilterChain ${chain.id}] Status is now: ${addFilterStatus}`);
    }, [addFilterStatus, chain.id]);

  const [implied, setImplied] = useState(false);
  const [marker, setMarker] = useState(null);
  const [filterStateId, setFilterStateId] = useState(null); // Unique ID to track filter state changes

  const debouncedFilters = useDebounce(chain.filters, 300); // Debounce filters
  const { hasChanges, storedFilterValues, setStoredFilterValues } = useFilterChainState(
    { ...chain, filters: debouncedFilters }, // Use debounced filters for change detection
    addFilterStatus, 
    implied, 
    setImplied, 
    isMain, 
    filterStateId
  );

  const [storedStateId, setStoredStateId] = useState(null); // Stored ID to compare changes
  const [storedSourceStateId, setStoredSourceStateId] = useState(null); // Stored source ID to compare changes
  const [isActual, setIsActual] = useState(true);
  const loadedStateIdRef = useRef(null);
  const [title, setTitle] = useState(loadedChainName || (staticLabel ? accordionLabel : `NowyFiltr-${chainIndex + 1}`));

  //console.log('FilterChainAccordion2 staticLabel:', staticLabel);
  //console.log('filterStateId', filterStateId,'IsActual:', isActual);

  //console.log('LoadCheck - ', 'isActual', isActual, 'hasChanges', hasChanges, 'addFilterStatus', addFilterStatus, 'implied', implied);

  //const accordionTitle = useMemo(() => generateAccordionTitle(chain.filters, chainIndex), [chain.filters, chainIndex]);
  const summaryParts = useMemo(() => getAccordionSummaryParts(chain.filters), [chain.filters]);
  const indicator = useMemo(() => getIndicatorColor(hasChanges, addFilterStatus, isActual), [hasChanges, addFilterStatus, isActual]);

  const { handleAddOrUpdate, handleStop, setIsStopped } = useFilterQuery({
    filters: chain.filters,
    status: debouncedAddFilterStatus,
    onStatusChange: setAddFilterStatus,
    implied,
    onImpliedChange: setImplied,
    mapRef,
    accordionSummary: title,
    marker,
    setMarker,
    endpoint: calculation_endpoint,
    filterStateId,
    setFilterStateId,
    stateId,
    setStoredStateId,
    fitBounds,
    allFilterStateIds
  });

  const bulkUpdateTimeoutRef = useRef(null);

  useEffect(() => {
    if (onBulkUpdate && onBulkUpdate.id !== undefined && onBulkUpdate.id !== lastProcessedBulkUpdate) {
      if (indicator.status === 'warning') {
        // Stagger to avoid simultaneous session writes (race causing lost query_ids)
        const delayMs = chainIndex * 250; // tune (e.g. 150–300)
        bulkUpdateTimeoutRef.current = setTimeout(() => {
          handleAddOrUpdate();
        }, delayMs);
      }
      setLastProcessedBulkUpdate(onBulkUpdate.id);
    }
  }, [
    onBulkUpdate,
    lastProcessedBulkUpdate,
    indicator.status,
    handleAddOrUpdate,
    chainIndex
  ]);

  useEffect(() => {
    return () => {
      if (bulkUpdateTimeoutRef.current) {
        clearTimeout(bulkUpdateTimeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (onBulkStop && onBulkStop.id !== undefined && onBulkStop.id !== lastProcessedBulkStop) {
      // Use internal status to ensure we catch real in-flight queries
      if (addFilterStatus === 'stop') {
        console.log(`[FilterChain ${chain.id}] Bulk stop received (id=${onBulkStop.id}). Stopping.`);
        handleStop(true);
      } else {
        console.log(`[FilterChain ${chain.id}] Bulk stop ignored; current status=${addFilterStatus}, visual=${indicator.status}`);
      }
      setLastProcessedBulkStop(onBulkStop.id);
    }
  }, [
    onBulkStop,
    lastProcessedBulkStop,
    addFilterStatus,
    indicator.status,
    handleStop
  ]);

  useEffect(() => {
    if (onMarkerCreated && marker) {
      onMarkerCreated(chain.id, marker);
    }
  }, [marker, onMarkerCreated, chain.id]);


  useEffect(() => {
    const loadedId = chain.loadedFilterStateId;
    const loadedStateId = chain.loadedStateId;
    //console.log('Loaded filterStateId:', loadedId, 'Current ref:', loadedStateIdRef.current);
    if (loadedId && loadedId !== loadedStateIdRef.current) {
      loadedStateIdRef.current = loadedId;
      setFilterStateId(loadedId);
      setStoredStateId(loadedStateId);
      setStoredFilterValues(null); // Clear stored values to force re-check
      setImplied(false);
      setAddFilterStatus('update');
      setTitle(chain.loadedTitle || title);
      setStoredSourceStateId(chain.loadedSourceStateId || null);
      setSourceStateId(chain.loadedSourceStateId || null);
    }
  }, [chain.loadedFilterStateId]);

  const handleRestoreValues = () => {
    if (storedFilterValues) {
      const restoredFilters = chain.filters.map(filter => {
        const stored = storedFilterValues.find(f => f.id === filter.id);
        return { ...filter, selectedValue: stored ? stored.selectedValue : null };
      });

      onToggle(chain.id, chain.isExpanded, { ...chain, filters: restoredFilters });
    }
  };

  useEffect(() => {
    if (onStateChange) {
      console.log(`[FilterChain ${chain.id}] Firing onStateChange. Status: ${addFilterStatus} (visual: ${indicator.status})`);
      onStateChange(chain.id, {
        storedFilterValues,
        filterStateId,
        storedStateId,
        title,
        sourceStateId,
        storedSourceStateId,
        status: addFilterStatus,        // internal machine status
        visualStatus: indicator.status, // outward UI status
      });
    }
  }, [
    chain.id,
    storedFilterValues,
    filterStateId,
    storedStateId,
    title,
    onStateChange,
    sourceStateId,
    storedSourceStateId,
    addFilterStatus,
    indicator.status
  ]);

  useEffect(() => {
    if (filterStateId && isMain) {
        setStoredStateId(filterStateId);
        setStateId(filterStateId);
    }
  }, [filterStateId, setStateId, isMain]);
  
  useEffect(() => {
    if (stateId && filterStateId) {
      setIsActual(stateId === storedStateId);
    }
  }, [stateId, filterStateId, storedStateId]);
  
  useEffect(() => {
    if (filterStateId && !isTarget) {
        setSourceStateId(filterStateId);
    }
    if (filterStateId && isTarget) {
        setStoredSourceStateId(sourceStateId);
    }
  }, [filterStateId, setSourceStateId, isTarget]);

  useEffect(() => {
    if (isTarget && sourceStateId && filterStateId) {
      setIsActual(sourceStateId === storedSourceStateId);
    }
  }, [sourceStateId, filterStateId, storedSourceStateId, isTarget]);

  const memoizedHasChanges = useMemo(() => hasChanges, [hasChanges]);

  const hasNonPassiveFilterWithoutChildren = chain.filters.some(
    (filter) => !filter.children && !filter.ispassive
  );

  const handleAccordionToggle = useCallback((event, expanded) => {
    onToggle(chain.id, expanded, { ...chain, isExpanded: expanded });
  }, [onToggle, chain]);

  const handleSummaryToggle = useCallback((e) => {
    e.stopPropagation();
    onToggle(chain.id, !chain.isExpanded, { ...chain, isExpanded: !chain.isExpanded });
  }, [onToggle, chain]);

  return (
    <Box sx={{ position: 'relative' }}>
    {/* Top border for first accordion as a separate element */}
    {chainIndex === 0 && !staticLabel && (
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: '20px',
          right: '20px',
          height: '0.5px',
          backgroundColor: 'rgb(160, 160, 160)',
          zIndex: 1,
        }}
      />
    )}
    <Accordion
      expanded={chain.isExpanded}
      onChange={handleAccordionToggle}
      disableGutters
      sx={{
        ...(staticLabel
          ? {
              boxShadow: 'none',
              '&.MuiAccordion-root': { border: 'none', '&:before': { display: 'none' } },
            }
          : {
              mt: 0,
              mb: 0,
              boxShadow: 'none',
              position: 'relative',
              borderRadius: 0,
              // Hide MUI's default :before for all accordions
              '&.MuiAccordion-root:before': {
                display: 'none',
              },
              // Bottom border for all accordions using :after
              '&:after': {
                content: '""',
                position: 'absolute',
                bottom: 0,
                left: '20px',
                right: '20px',
                height: '0.5px',
                backgroundColor: 'rgb(160, 160, 160)',
                zIndex: 1,
              },
            }),
      }}
    >
        <AccordionSummary
          sx={{
            ...(staticLabel
              ? {
                  minHeight: '40px !important',
                  height: '40px !important',
                  '& .MuiAccordionSummary-content': { m: '0 !important', flexGrow: 1 },
                  '& .MuiAccordionSummary-root': { p: '0 !important' },
                  pr: '4px',
                }
              : {
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
                  '& .MuiAccordionSummary-root': { padding: '0 !important' },
                  pr: '4px',
                }),
          }}
        >
          <AccordionSummaryContent
            summaryParts={summaryParts}
            chain={chain}
            marker={marker}
            mapRef={mapRef}
            variant={labelVariant}
            isStatic={staticLabel}
            title={title}
            setTitle={setTitle}
            onToggle={handleSummaryToggle}
            isExpanded={chain.isExpanded}
            isLayerVisible={isLayerVisible}
            setLayerVisible={setLayerVisible}
            statusButton={
              <AccordionStatusButton
                indicator={indicator}
                handleAddOrUpdate={handleAddOrUpdate}
                handleStop={handleStop}
                isExpanded={chain.isExpanded}
              />
            }
          />
        </AccordionSummary>
        <AccordionDetails sx={{ p: 0 }}>
          {chain.filters.map((filter) => (
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
                  {renderFilterComponent(chain.id, { ...filter, type: filter.type, addFilterStatus, disabled: addFilterStatus === 'stop'  })}
                </Box>
              </Box>
            </Box>
          ))}
          {chain.filters.length > 0 && hasNonPassiveFilterWithoutChildren && (
            <FilterChainButtons
              hasChanges={hasChanges}
              handleRestoreValues={handleRestoreValues}
              onRemove={onRemove}
              chainId={chain.id}
              marker={marker}
              showDeleteButton={showDeleteButton}
              handleAddOrUpdate={handleAddOrUpdate}
              handleStop={handleStop}
              filters={chain.filters}
              status={addFilterStatus}
              isActual={isActual}
              isMain={isMain}
              stateId={stateId}
            />
          )}
        </AccordionDetails>
        {renderBottomBar(addFilterStatus)}
      </Accordion>
    </Box>
  );
}

export default React.memo(FilterChainAccordion2);