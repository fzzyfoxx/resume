// src/components/sidebar/MainFilterAccordion.jsx
import React from 'react';
import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Typography,
  Box,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import CircularLoader from '../common/CircularLoader'; // Using the new loader component

/**
 * Renders the main 'Filters' accordion.
 * @param {object} props - Component props.
 * @param {boolean} props.expanded - Whether the accordion is expanded.
 * @param {function} props.onToggle - Callback for accordion expansion.
 * @param {boolean} props.isLoading - Whether the initial filters are loading.
 * @param {React.Node} props.children - Child elements (the filter chains and "Add New Filter Chain" button).
 */
function MainFilterAccordion({caption, expanded, onToggle, isLoading, children }) {
  return (
    <Accordion
      expanded={expanded}
      onChange={onToggle}
      disableGutters
      sx={{
        boxShadow: 'none',
        '&.MuiAccordion-root': {
          border: 'none',
          '&:before': {
            display: 'none',
          },
        },
      }}
    >
      <AccordionSummary expandIcon={<ExpandMoreIcon />}
        sx={{
          minHeight: '40px !important',
          height: '40px !important',
          '& .MuiAccordionSummary-content': {
            margin: '0 !important',
            flexGrow: 1,
          },
          '& .MuiAccordionSummary-root': {
            padding: '0 !important',
          },
          pr: 1
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', ml: 1.5, py: 0.75 }}>
          <Typography variant="h8" color="textSecondary">{caption}</Typography>
          {isLoading && (
            <CircularLoader size={16} sx={{ ml: 1 }} />
          )}
        </Box>
      </AccordionSummary>
      <AccordionDetails sx={{ p: 0}}>
            {children}
        </AccordionDetails>
    </Accordion>
  );
}

export default MainFilterAccordion;