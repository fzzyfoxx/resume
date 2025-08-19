// src/components/sidebar/FilterChainAccordion.jsx
import React from 'react';
import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Typography,
  Box,
  Divider, // Still imported but its use case might change based on parent
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import CircularLoader from '../common/CircularLoader';
import AddButton from './AddButton';

/**
 * Renders a single filter chain accordion.
 * @param {object} props - Component props.
 * @param {object} props.chain - The filter chain object.
 * @param {number} props.chainIndex - The index of the chain in the list.
 * @param {function} props.onToggle - Callback for accordion expansion.
 * @param {function} props.renderFilterComponent - Function passed from parent to render individual ComboBoxFilters/MultiSelectFilters.
 */
function FilterChainAccordion({ chain, chainIndex, onToggle, renderFilterComponent }) {
  // Function to handle value changes in filters
  const handleFilterValueChange = (filterId, value, symbols, hasChildren, isValueEmpty, filterType) => {
    const updatedFilters = chain.filters.map((filter) => {
      if (filter.id === filterId) {
        return {
          ...filter,
          selectedValue: isValueEmpty ? null : value, // Update selectedValue based on isValueEmpty
        };
      }
      return filter;
    });
  
    const updatedChain = {
      ...chain,
      filters: updatedFilters,
    };
  
    // Update the accordion title based on the updated filters
    const selectedFilterParts = updatedChain.filters.map((f) => {
      if (!f.selectedValue || (Array.isArray(f.selectedValue) && f.selectedValue.length === 0)) {
        return null;
      }
      // Use the selected value for ComboBoxFilter, otherwise use the filter's title
      return f.selector_type === 'combo_box' ? f.selectedValue : f.title;
    }).filter(Boolean);
  
    const newAccordionTitle = selectedFilterParts.length > 0
      ? selectedFilterParts.join(' > ')
      : `Filter Chain ${chainIndex + 1}`;
  
    // Update the chain object and accordion title
    onToggle(updatedChain.id, updatedChain.isExpanded, newAccordionTitle);
  };

  // Memoized accordion title logic
  const accordionTitle = React.useMemo(() => {
    const selectedFilterParts = chain.filters.map((f) => {
      console.log(`[FilterChainAccordion] Filter ID: ${f.id}, Type: ${f.type}, SelectedValue: ${f.selectedValue}, Title: ${f.title}`); // Add this line
      if (!f.selectedValue || (Array.isArray(f.selectedValue) && f.selectedValue.length === 0)) {
        return null;
      }
      return f.selector_type === 'combo_box' ? f.selectedValue : f.title;
    }).filter(Boolean);

    if (selectedFilterParts.length > 0) {
      return selectedFilterParts.join(' > ');
    }
    return `Filter Chain ${chainIndex + 1}`;
  }, [chain.filters, chainIndex]);

  return (
    <Accordion
      expanded={chain.isExpanded}
      onChange={(event, expanded) => onToggle(chain.id, expanded)}
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
      <AccordionSummary expandIcon={<ExpandMoreIcon />}
        sx={{
          minHeight: '36px !important',
          alignItems: 'center',
          '& .MuiAccordionSummary-content': {
            margin: '0 !important',
            flexGrow: 1,
            display: 'block',
          },
          '& .MuiAccordionSummary-root': {
            padding: '0 !important',
          },
          pr: 1,
        }}
      >
        <Box sx={{
          display: 'flex',
          alignItems: 'center',
          ml: 1.5,
          py: 0.5,
          minHeight: '26px',
        }}>
          <Typography
            variant="caption"
            color="textSecondary"
            sx={{
              flexGrow: 1,
              wordBreak: 'break-word',
            }}
          >
            {accordionTitle}
          </Typography>
          {chain.isLoading && (
            <CircularLoader size={14} sx={{ ml: 0.75, flexShrink: 0 }} />
          )}
        </Box>
      </AccordionSummary>
      <AccordionDetails sx={{ p: 0 }}>
      {chain.filters.map((filter) => (
        <Box key={filter.id} sx={{ mb: 0.5 }}>
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'center',
              my: 0.5,
              px: 1,
            }}
          >
            <Box sx={{ width: '80%', maxWidth: 'calc(100% - 24px)' }}>
              {renderFilterComponent(chain.id, { ...filter, type: filter.type }, (value, symbols, hasChildren, isValueEmpty) =>
                handleFilterValueChange(filter.id, value, symbols, hasChildren, isValueEmpty, filter.type)
              )}
            </Box>
          </Box>
        </Box>
      ))}
        {chain.filters.length > 0 && !chain.filters[chain.filters.length - 1].children && (
          <AddButton onClick={() => console.log(`'ADD' button clicked for finished filter chain. No action implemented.`)} />
        )}
      </AccordionDetails>
    </Accordion>
  );
}

export default FilterChainAccordion;