import React, {useState, useMemo, useEffect, useRef} from 'react';
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
    onStateChange
    }) {
  const [addFilterStatus, setAddFilterStatus] = useState('add');
  const [implied, setImplied] = useState(false);
  const [marker, setMarker] = useState(null);
  const [filterStateId, setFilterStateId] = useState(null); // Unique ID to track filter state changes
  const { hasChanges, storedFilterValues, setStoredFilterValues } = useFilterChainState(chain, addFilterStatus, implied, setImplied, isMain, filterStateId);
  const [storedStateId, setStoredStateId] = useState(null); // Stored ID to compare changes
  const [isActual, setIsActual] = useState(true);
  const loadedStateIdRef = useRef(null);
  const [title, setTitle] = useState(staticLabel ? accordionLabel : `NowyFiltr-${chainIndex + 1}`);

  console.log('FilterChainAccordion2 staticLabel:', staticLabel);
  console.log('filterStateId', filterStateId,'IsActual:', isActual);

  //console.log('LoadCheck - ', 'isActual', isActual, 'hasChanges', hasChanges, 'addFilterStatus', addFilterStatus, 'implied', implied);

  const accordionTitle = useMemo(() => generateAccordionTitle(chain.filters, chainIndex), [chain.filters, chainIndex]);
  const summaryParts = useMemo(() => getAccordionSummaryParts(chain.filters), [chain.filters]);
  const indicator = useMemo(() => getIndicatorColor(hasChanges, addFilterStatus, isActual), [hasChanges, addFilterStatus, isActual]);

  const { handleAddOrUpdate, handleStop } = useFilterQuery({
    filters: chain.filters,
    status: addFilterStatus,
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
  });

  useEffect(() => {
    console.log('TempFilter - filter status', addFilterStatus);
  }, [addFilterStatus]);

  useEffect(() => {
    const loadedId = chain.loadedFilterStateId;
    const loadedStateId = chain.loadedStateId;
    console.log('Loaded filterStateId:', loadedId, 'Current ref:', loadedStateIdRef.current);
    if (loadedId && loadedId !== loadedStateIdRef.current) {
      loadedStateIdRef.current = loadedId;
      setFilterStateId(loadedId);
      setStoredStateId(loadedStateId);
      setStoredFilterValues(null); // Clear stored values to force re-check
      setImplied(false);
      setAddFilterStatus('update');
      setTitle(chain.loadedTitle || title);
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
    console.log('STATE UPDATE - filterStateId changed:', filterStateId, storedFilterValues, storedStateId);
    if (onStateChange) {
      onStateChange(chain.id, {
        storedFilterValues,
        filterStateId,
        storedStateId,
        title
      });
    }
  }, [storedFilterValues, title]);

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

  const memoizedHasChanges = useMemo(() => hasChanges, [hasChanges]);

  const hasNonPassiveFilterWithoutChildren = chain.filters.some(
    (filter) => !filter.children && !filter.ispassive
  );

  return (
    <Box sx={{ position: 'relative' }}>
      <Accordion
        expanded={chain.isExpanded}
        onChange={(event, expanded) => onToggle(chain.id, expanded, { ...chain, isExpanded: expanded })}
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
                borderTop: chainIndex === 0 ? '1px solid #eee' : 'none',
                borderBottom: '1px solid #eee',
                '&.MuiAccordion-root': { '&:before': { display: 'none' } },
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
            onToggle={(e) => {
              e.stopPropagation();
              onToggle(chain.id, !chain.isExpanded, { ...chain, isExpanded: !chain.isExpanded });
            }}
            isExpanded={chain.isExpanded}
            statusButton={
              <AccordionStatusButton
                indicator={indicator}
                handleAddOrUpdate={handleAddOrUpdate}
                handleStop={handleStop}
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
                  {renderFilterComponent(chain.id, { ...filter, type: filter.type, addFilterStatus })}
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

export default FilterChainAccordion2;