// src/components/sidebar/FilterChainAccordion.jsx
import React, { useState, useEffect } from 'react';
import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Typography,
  Box,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import CircularLoader from '../common/CircularLoader';
import AddButton from './AddButton';
import Subtitle from '../common/Subtitle';
import QualificationFilter from '../filters/QualificationFilter';

function FilterChainAccordion({ chain, chainIndex, onToggle, renderFilterComponent }) {
  // State to hold the QualificationFilter's value
  // Initialize with any existing qualification data from the chain if available
  // You might need to adjust how your 'chain' object stores this qualification data
  const [qualificationValue, setQualificationValue] = useState(chain.qualificationFilterValue || null);

  useEffect(() => {
    // This effect ensures that if the parent chain object's qualificationFilterValue changes,
    // the internal state is updated.
    setQualificationValue(chain.qualificationFilterValue || null);
  }, [chain.qualificationFilterValue]); // Depend on qualificationFilterValue from chain

  useEffect(() => {
    const resetQualificationValueIfChildrenExist = () => {
      if (chain.filters[chain.filters.length - 1]?.children) {
        setQualificationValue(null);
      }
    };
    const SetInitialQualificationValue = () => {
      // Set initial qualification value when the component mounts or chain.filters changes
      if (!qualificationValue && !chain.qualificationFilterValue) {
        setQualificationValue({option: 'Excluded', label: 'obszar wykluczony', value: ''});
      }
    };
    SetInitialQualificationValue();
    resetQualificationValueIfChildrenExist();
  }, [chain.filters]);

  const handleFilterValueChange = (filterId, value, symbols, hasChildren, isValueEmpty, filterType) => {
    if (filterType === 'qualification') {
      // Handle QualificationFilter's value change
      const newQualificationValue = isValueEmpty ? null : value;
      setQualificationValue(newQualificationValue);

      // Also, update the chain prop with the new qualification value if you want it persistent
      // This part depends on how you want to manage chain state in the parent component
      // For now, let's assume `onToggle` can take an updated chain object
      onToggle(
        chain.id,
        chain.isExpanded,
        { ...chain, qualificationFilterValue: newQualificationValue } // Pass updated chain with qualification
      );
    } else {
      // Handle other filter types
      const updatedFilters = chain.filters.map((filter) => {
        if (filter.id === filterId) {
          return {
            ...filter,
            selectedValue: isValueEmpty ? null : value,
          };
        }
        return filter;
      });

      const updatedChain = {
        ...chain,
        filters: updatedFilters,
      };

      // Call onToggle with the updated chain
      onToggle(updatedChain.id, updatedChain.isExpanded, updatedChain);
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

    // Add QualificationFilter label
    if (qualificationValue !== null) {
      selectedFilterParts.push(`${qualificationValue.label}`);
    }

    return selectedFilterParts.length > 0
      ? selectedFilterParts.join(' > ')
      : `New Filter ${chainIndex + 1}`;
  }, [chain.filters, chainIndex, qualificationValue]); // Add qualificationValue to dependencies

  return (
    <Accordion
      expanded={chain.isExpanded}
      onChange={(event, expanded) => onToggle(chain.id, expanded, { ...chain, isExpanded: expanded })} // Pass updated chain on toggle
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
            display: 'block',
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
            alignItems: 'center',
            ml: 1.5,
            py: 0.5,
            minHeight: '26px',
          }}
        >
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
              marginBottom: '10px',
              marginTop: '10px',
            }}
          >
            <Subtitle label="Objects definition" />
          </Box>
        )}
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
                {renderFilterComponent(chain.id, { ...filter, type: filter.type }, (value, symbols, hasChildren, isValueEmpty) =>
                  handleFilterValueChange(filter.id, value, symbols, hasChildren, isValueEmpty, filter.type)
                )}
              </Box>
            </Box>
          </Box>
        ))}
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
                marginBottom: '10px',
                marginTop: '10px',
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
                defaultValue={qualificationValue} // Pass the state as defaultValue
                compact={true}
                onValueChange={(filterId, value, isValueEmpty) =>
                  handleFilterValueChange(filterId, value, null, null, isValueEmpty, 'qualification')
                }
              />
            </Box>
          </>
        )}
      </AccordionDetails>
    </Accordion>
  );
}

export default FilterChainAccordion;