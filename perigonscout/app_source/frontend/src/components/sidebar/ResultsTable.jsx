import React, { useState, useMemo, useRef } from 'react';
import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  TablePagination,
  Paper,
  IconButton,
  Tooltip,
  Typography,
  Popper,
  Fade,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import MyLocationIcon from '@mui/icons-material/MyLocation';
import L from 'leaflet';
import { mainColor, lightIconColor, disabledIconColor, defaultIconColor } from '../../styles/ButtonStyles';

const headCellSx = {
  fontWeight: 600,
  backgroundColor: mainColor,
  whiteSpace: 'normal',
  fontSize: '0.6rem',
  color: 'white',
  padding: '12px 6px',
  lineHeight: 1.1,
  textAlign: 'right',
  verticalAlign: 'bottom',
  '& .MuiTableSortLabel-root': {
    display: 'inline-flex',
    flexDirection: 'row-reverse',   // keep arrow on the left
    alignItems: 'bottom',           // was flex-start -> now centers arrow vertically
    whiteSpace: 'normal',
    lineHeight: 1.1,
  },
  '& .MuiTableSortLabel-root:hover, & .MuiTableSortLabel-root.Mui-active': {
    color: 'inherit',
  },
  '& .MuiTableSortLabel-icon': {
    fontSize: '0.85rem',
    marginRight: 1.5,
    marginLeft: 0,
    alignSelf: 'bottom',            // was flex-start
    color: 'white !important',
  },
  '&:hover': {
    color: lightIconColor,
    '& .MuiTableSortLabel-icon': {
      color: `${lightIconColor} !important`,
    },
  },
};

const bodyCellSx = {
    fontSize: '0.65rem',
    whiteSpace: 'normal',
    wordBreak: 'break-word',
    padding: '4px 6px',
    lineHeight: 1.2,
    verticalAlign: 'center',
    textAlign: 'right',
    maxWidth: 240,
};

const integerFormatter = new Intl.NumberFormat('en-US', {
  useGrouping: true,
  maximumFractionDigits: 0,
});

const numericFormatter = new Intl.NumberFormat('en-US', {
  useGrouping: true,
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

function descendingComparator(a, b, orderBy, isNumeric) {
  let valA = a[orderBy];
  let valB = b[orderBy];

  if (isNumeric) {
    valA = parseFloat(valA);
    valB = parseFloat(valB);
    if (isNaN(valA)) valA = -Infinity;
    if (isNaN(valB)) valB = -Infinity;
  }

  if (valB < valA) {
    return -1;
  }
  if (valB > valA) {
    return 1;
  }
  return 0;
}

function getComparator(order, orderBy, columns) {
  const column = columns.find(c => c.id === orderBy);
  const isNumeric = column && (column.type === 'numeric' || column.type === 'integer');
  return order === 'desc'
    ? (a, b) => descendingComparator(a, b, orderBy, isNumeric)
    : (a, b) => -descendingComparator(a, b, orderBy, isNumeric);
}

function stableSort(array, comparator) {
  const stabilizedThis = array.map((el, index) => [el, index]);
  stabilizedThis.sort((a, b) => {
    const order = comparator(a[0], b[0]);
    if (order !== 0) {
      return order;
    }
    return a[1] - b[1];
  });
  return stabilizedThis.map((el) => el[0]);
}

const ResultsTable = ({ results, onSelectFeature, onShowInfo, mapRef }) => {
  const [order, setOrder] = useState('asc');
  const [orderBy, setOrderBy] = useState('');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  
  const [anchorEl, setAnchorEl] = useState(null);
  const [isPopperOpen, setIsPopperOpen] = useState(false);
  const popperTimeoutRef = useRef(null);
  const [expanded, setExpanded] = useState(true);

  const handleMouseEnter = (event) => {
    if (results && results.length > 0) {
      clearTimeout(popperTimeoutRef.current);
      setAnchorEl(event.currentTarget);
      setIsPopperOpen(true);
    }
  };

  const handleMouseLeave = () => {
    popperTimeoutRef.current = setTimeout(() => {
      setIsPopperOpen(false);
    }, 200); // Delay to allow moving cursor into the popper
  };

  const handlePopperMouseEnter = () => {
    clearTimeout(popperTimeoutRef.current);
  };

  const handleRequestSort = (event, property) => {
    // If switching to a new column, start with descending
    if (orderBy !== property) {
      setOrder('desc');
      setOrderBy(property);
      return;
    }
    // Same column: just toggle
    setOrder(prev => (prev === 'desc' ? 'asc' : 'desc'));
  };

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleGoToFeature = (feature) => {
    if (mapRef.current && feature.geometry) {
        const geoJsonLayer = L.geoJSON(feature);
        mapRef.current.fitBounds(geoJsonLayer.getBounds());
    }
    // This will trigger the highlight effect in MapComponent
    onSelectFeature(feature);
    // This will show the info box
    onShowInfo([feature]);
  };

  const { columns, rows } = useMemo(() => {
    if (!results || results.length === 0) {
      return { columns: [], rows: [] };
    }
    const firstFeature = results[0];
    const cols = firstFeature.properties.columns
      ? firstFeature.properties.columns
          .filter(c => c.type !== 'title')
          .map(c => ({ id: c.name, label: c.title, type: c.type })) // Include type
      : Object.keys(firstFeature.properties).map(p => ({ id: p, label: p, type: 'string' }));

    const tableRows = results.map((feature, index) => {
        const row = { id: feature.id || index, feature: feature };
        cols.forEach(c => {
            row[c.id] = feature.properties[c.id];
        });
        return row;
    });

    return { columns: cols, rows: tableRows };
  }, [results]);

  const formatCell = (value, type) => {
    if (value === null || typeof value === 'undefined') {
      return '';
    }
    if (type === 'numeric') {
      const numValue = parseFloat(
        (typeof value === 'string')
          ? value.replace(/\s/g, '').replace(/,/g, '.')
          : value
      );
      if (isNaN(numValue)) return String(value);
      // en-US gives: 12,345.67 -> change commas to spaces => 12 345.67
      return numericFormatter.format(numValue).replace(/,/g, ' ');
    }
    if (type === 'integer') {
      const intValue = parseInt(
        (typeof value === 'string')
          ? value.replace(/\s/g, '').replace(/,/g, '.')
          : value,
        10
      );
      if (isNaN(intValue)) return String(value);
      return integerFormatter.format(intValue).replace(/,/g, ' ');
    }
    if (type === 'multiBox') {
      let arrayValue = value;
      if (typeof arrayValue === 'string') {
        try {
          const jsonString = arrayValue.replace(/'/g, '"');
          arrayValue = JSON.parse(jsonString);
        } catch (e) {
          return String(value);
        }
      }
      if (!Array.isArray(arrayValue)) {
        return String(value);
      }
      return (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.25 }}>
          {arrayValue.map((item, index) => (
            <Box
              key={index}
              title={item}
              sx={{
                backgroundColor: 'rgba(0,0,0,0.08)',
                borderRadius: '3px',
                px: 0.5,
                py: 0.25,
                fontSize: '0.55rem',
                lineHeight: 1.1,
                display: 'block',
                width: '100%',
                whiteSpace: 'nowrap',        // prevent breaking inside item
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}
            >
              {item}
            </Box>
          ))}
        </Box>
      );
    }
    return String(value);
  };

  const hasResults = rows.length > 0;

  const visibleRows = stableSort(rows, getComparator(order, orderBy, columns)).slice(
    page * rowsPerPage,
    page * rowsPerPage + rowsPerPage,
  );

  const tableContent = (
    <>
      <TableContainer sx={{ maxHeight: 440, width: 'fit-content', maxWidth: '100%' }}>
        <Table
          stickyHeader
          size="small"
          aria-label="results table"
          sx={{
            tableLayout: 'auto',
            width: 'auto',
            maxWidth: '100%',
            '& th, & td': {
              width: '1%',
            },
            // Add right padding space that inherits header cell background
            '& thead th:last-of-type': {
              paddingRight: '24px',   // adjust as needed
            },
            // Keep body aligned with header (optional; remove if you want only header padded)
            '& tbody td:last-of-type': {
              paddingRight: '24px',
            },
          }}
        >
          <TableHead>
            <TableRow>
              <TableCell
                sx={{
                  ...headCellSx,
                  width: '1%',
                  padding: '4px 4px',
                }}
              />
              {columns.map((headCell) => {
                const wrappedLabel = headCell.label
                  ? headCell.label.split(/\s+/).map((w, i, arr) => (
                      <React.Fragment key={i}>
                        {w}
                        {i < arr.length - 1 && <br />
                        }
                      </React.Fragment>
                    ))
                  : headCell.id;
                return (
                  <TableCell
                    key={headCell.id}
                    align="left"            // changed from center
                    padding="normal"
                    sortDirection={orderBy === headCell.id ? order : false}
                    sx={{
                      ...headCellSx,
                      width: '1%',
                      maxWidth: 160,
                    }}
                  >
                    <TableSortLabel
                      active={orderBy === headCell.id}
                      direction={orderBy === headCell.id ? order : 'asc'}
                      onClick={(e) => handleRequestSort(e, headCell.id)}
                    >
                      {wrappedLabel}
                    </TableSortLabel>
                  </TableCell>
                );
              })}
            </TableRow>
          </TableHead>
          <TableBody>
            {visibleRows.map((row) => (
              <TableRow hover tabIndex={-1} key={row.id}>
                <TableCell
                  padding="none"
                  align="center"
                  sx={{
                    padding: '4px 6px 4px 10px', // left padding added
                    width: '1%',
                  }}
                >
                  <Tooltip title="znajdź" placement="left">
                    <IconButton
                      size="small"
                      onClick={() => handleGoToFeature(row.feature)}
                      sx={{
                        color: disabledIconColor,
                        '&:hover': { color: defaultIconColor },
                      }}
                    >
                      <MyLocationIcon fontSize="inherit" />
                    </IconButton>
                  </Tooltip>
                </TableCell>
                {columns.map((column) => (
                  <TableCell key={column.id} align="left" sx={bodyCellSx}>
                    {formatCell(row[column.id], column.type)}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      <TablePagination
        rowsPerPageOptions={[5, 10, 25]}
        component="div"
        count={rows.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={handleChangePage}
        onRowsPerPageChange={handleChangeRowsPerPage}
        labelRowsPerPage="Wierszy na stronę:"
        labelDisplayedRows={({ count, page }) => {
          const totalPages = count > 0 ? Math.ceil(count / rowsPerPage) : 1;
          return `Strona ${page + 1} z ${totalPages}`;
        }}

        SelectProps={{
          MenuProps: {
            PaperProps: {
              sx: {
                '& .MuiMenuItem-root': {
                  fontSize: '0.7rem',
                  minHeight: 'auto',
                  py: 0.4,
                  lineHeight: 1.1,
                },
              },
            },
          },
        }}
        sx={{
            my: 1.5,
          '& .MuiTablePagination-toolbar': {
            minHeight: 28,
            px: 0.5,
          },
          '& .MuiTablePagination-selectLabel, & .MuiTablePagination-displayedRows': {
            fontSize: '0.7rem',
            lineHeight: 1.1,
          },
            '& .MuiTablePagination-select': {
            fontSize: '0.7rem',
            lineHeight: 1.1,
            py: 0.25,
          },
          '& .MuiTablePagination-actions': {
            '& button': { p: 0.25 },
          },
          '& .MuiSelect-icon': {
            fontSize: '0.8rem',
          },
        }}
      />
    </>
  );

  return (
    <Accordion
      expanded={expanded}
      onChange={() => setExpanded(!expanded)}
      disableGutters
      sx={{
        boxShadow: 'none',
        '&.MuiAccordion-root': { border: 'none', '&:before': { display: 'none' } },
      }}
    >
      <AccordionSummary
        expandIcon={<ExpandMoreIcon />}
        aria-controls="results-panel-content"
        id="results-panel-header"
        sx={{
          minHeight: '40px !important',
          height: '40px !important',
          padding: 0,
          pr: '4px',
          '& .MuiAccordionSummary-content': { m: '0 !important', alignItems: 'center' },
        }}
      >
        <Box sx={{ flexGrow: 1, ml: '20px' }}>
          <Typography variant="h6" color="textSecondary" sx={{ fontSize: '0.875rem', fontWeight: 600 }}>
            Wyniki
          </Typography>
        </Box>
      </AccordionSummary>
      <AccordionDetails sx={{ p: 0, pt: 1 }}>
        <Box sx={{ paddingLeft: '18px', paddingRight: '22px' }}>
          <Paper
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
            ref={setAnchorEl} // Set anchor for the popper
            sx={{
              paddingLeft: '16px',
              paddingRight: '20px',
              cursor: hasResults ? 'pointer' : 'default',
              backgroundColor: 'transparent',
              boxShadow: 'none',
              border: '1px dashed',
              borderColor: hasResults ? 'grey.500' : 'grey.300',
              textAlign: 'center',
              height: '40px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              '&:hover': {
                borderColor: hasResults ? 'text.primary' : 'grey.300',
                backgroundColor: hasResults ? 'rgba(0, 0, 0, 0.04)' : 'transparent',
              },
              transition: 'background-color 0.3s, border-color 0.3s',
            }}
          >
            <Typography variant="caption" color="textSecondary" sx={{ fontSize: '0.75rem', py: 1 }}>
              {hasResults ? `pokaż szczegółowe wyniki` : 'brak wyników do wyświetlenia.'}
            </Typography>
          </Paper>

          {hasResults && (
            <Popper
              open={isPopperOpen}
              anchorEl={anchorEl}
              placement="right-start"
              transition
              sx={{ zIndex: 1300 }} // High z-index to be on top of everything
              onMouseEnter={handlePopperMouseEnter}
              onMouseLeave={handleMouseLeave}
            >
              {({ TransitionProps }) => (
                <Fade {...TransitionProps} timeout={350}>
                  <Paper
                    elevation={6}
                    sx={{
                      width: 'fit-content',
                      maxWidth: '85vw',
                      maxHeight: '70vh',
                      overflow: 'auto',
                      ml: 1,
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      backdropFilter: 'blur(8px)',
                      p: 0,          // removed padding around table
                    }}
                  >
                    {tableContent}
                  </Paper>
                </Fade>
              )}
            </Popper>
          )}
        </Box>
      </AccordionDetails>
    </Accordion>
  );
};

export default ResultsTable;