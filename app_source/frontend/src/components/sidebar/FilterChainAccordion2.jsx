import React from 'react';
import { Accordion, AccordionSummary, AccordionDetails, Box, Typography, IconButton, Tooltip } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { useFilterChainState } from '../../hooks/checkChanges';
import { generateAccordionTitle, getIndicatorColor } from '../../utils/filterChainUtils';
import { renderBottomBar } from '../common/horizontalLoader';
import FilterChainButtons from './bottomButtons';
import AccordionSummaryContent from './AccordionSummaryContent';

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
    }) {
  const [addFilterStatus, setAddFilterStatus] = React.useState('initial');
  const [implied, setImplied] = React.useState(false);
  const [marker, setMarker] = React.useState(null);
  const { hasChanges, storedFilterValues, setStoredFilterValues } = useFilterChainState(chain, addFilterStatus);


  const accordionTitle = React.useMemo(() => generateAccordionTitle(chain.filters, chainIndex), [chain.filters, chainIndex]);
  const indicatorColor = React.useMemo(() => getIndicatorColor(hasChanges, addFilterStatus), [hasChanges, addFilterStatus]);

  const handleRestoreValues = () => {
    if (storedFilterValues) {
      const restoredFilters = chain.filters.map(filter => {
        const stored = storedFilterValues.find(f => f.id === filter.id);
        return { ...filter, selectedValue: stored ? stored.selectedValue : null };
      });

      onToggle(chain.id, chain.isExpanded, { ...chain, filters: restoredFilters });
    }
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
          backgroundColor: indicatorColor,
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
        <AccordionSummaryContent
            accordionTitle={accordionTitle}
            chain={chain}
            marker={marker}
            mapRef={mapRef}
            variant={labelVariant}
            label={accordionLabel}
            isStatic={staticLabel}
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
          {chain.filters.length > 0 && !chain.filters[chain.filters.length - 1].children && (
            <FilterChainButtons
              hasChanges={hasChanges}
              handleRestoreValues={handleRestoreValues}
              onRemove={onRemove}
              chainId={chain.id}
              marker={marker}
              filters={chain.filters}
              setAddFilterStatus={setAddFilterStatus}
              setImplied={setImplied}
              mapRef={mapRef}
              accordionSummary={accordionTitle}
              setMarker={setMarker}
              memoizedHasChanges={memoizedHasChanges}
              calculation_endpoint={calculation_endpoint}
              showDeleteButton={showDeleteButton}
            />
          )}
        </AccordionDetails>
        {renderBottomBar(addFilterStatus)}
      </Accordion>
    </Box>
  );
}

export default FilterChainAccordion2;