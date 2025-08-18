// src/components/sidebar/FilterChainAccordion.jsx
import React from 'react';
import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Typography,
  Box,
  Divider,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import CircularLoader from '../common/CircularLoader'; // Using the new loader component
import AddButton from './AddButton'; // The new ADD button component

/**
 * Renders a single filter chain accordion.
 * @param {object} props - Component props.
 * @param {object} props.chain - The filter chain object.
 * @param {number} props.chainIndex - The index of the chain in the list.
 * @param {function} props.onToggle - Callback for accordion expansion.
 * @param {function} props.renderFilterComponent - Function passed from parent to render individual ComboBoxFilters.
 */
function FilterChainAccordion({ chain, chainIndex, onToggle, renderFilterComponent }) {
  return (
    <Accordion
      expanded={chain.isExpanded}
      onChange={(event, expanded) => onToggle(chain.id, expanded)}
      disableGutters
      sx={{
        mt: 0,
        mb: 0,
        boxShadow: 'none',
        '&.MuiAccordion-root': {
          '&:before': {
            display: 'none',
          },
        },
      }}
    >
      <AccordionSummary expandIcon={<ExpandMoreIcon />}
        sx={{
          minHeight: '36px !important',
          height: '36px !important',
          '& .MuiAccordionSummary-content': {
            margin: '0 !important',
            flexGrow: 1,
          },
          '& .MuiAccordionSummary-root': {
            padding: '0 !important',
          },
          pr: 1,
          pb: 1.5
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', ml: 1.5, py: 0.5 }}>
          <Typography
            variant="caption"
            color="textSecondary"
          >
            {chain.filters.map(f => f.selectedValue).filter(Boolean).join(' > ') || `Filter Chain ${chainIndex + 1}`}
          </Typography>
          {chain.isLoading && (
              <CircularLoader size={14} sx={{ ml: 0.75 }} />
          )}
        </Box>
      </AccordionSummary>
      <AccordionDetails sx={{ p: 0 }}>
        {chain.filters.map((filter) => ( // Removed filterIndex as it's not used here for Divider
          <Box key={filter.id} sx={{ mb: 0.5 }}>
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'center',
                my: 0.5,
                px: 1
              }}
            >
              <Box sx={{ width: '90%', maxWidth: 'calc(100% - 24px)' }}>
                {renderFilterComponent(chain.id, filter)}
              </Box>
            </Box>
          </Box>
        ))}
        {/* ADD button below the last filter, aligned right */}
        {chain.filters.length > 0 && !chain.filters[chain.filters.length - 1].children && (
          <AddButton onClick={() => console.log(`'ADD' button clicked for finished filter chain. No action implemented.`)} />
        )}
      </AccordionDetails>
    </Accordion>
  );
}

export default FilterChainAccordion;