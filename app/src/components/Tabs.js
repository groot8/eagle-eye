import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Paper from '@material-ui/core/Paper';
import Tabs from '@material-ui/core/Tabs';
import Tab from '@material-ui/core/Tab';
import FixedContainer from './Container';
import FixedContainer2 from './Container2';
import { withRouter } from 'react-router-dom'
import { Route, Link, BrowserRouter ,Switch } from 'react-router-dom';


const useStyles = makeStyles({
  nav: {
    flexGrow: 1,
    width:840,
    marginTop: 20,
    
  },
  label: {
    color: '#009BE5',
    fontWeight:700,
    fontSize:16,
},
indicator: {
    backgroundColor: '#009BE5'
  }
});

export default function CenteredTabs() {
  const classes = useStyles();
  const [value, setValue] = React.useState(0);

  function handleChange(event, newValue) {
    setValue(newValue);
  }

  return (
    <BrowserRouter>
        
        <Paper className={classes.nav}>
        <Tabs
        
            value={value}
            onChange={handleChange}
            classes={{ indicator: classes.indicator }}
            indicatorColor='primary'
            centered
            
        >
            <Tab label="Stream" className={classes.label} component={Link} to="/feed"  />
            <Tab label="Map" className={classes.label} component={Link} to="/map" />
        </Tabs>

        </Paper>
        <Switch>
        <Route path="/feed" component={FixedContainer} />
        <Route path="/map" component={FixedContainer2} />
      </Switch>
        
    </BrowserRouter>
  );
}