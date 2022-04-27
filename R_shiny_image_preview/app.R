#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#
library(ggplot2)
library(shiny)
df_folders <- read.csv('../results/folder_table.dat', header = FALSE, sep = ' ', col.names = c('folder', 'name', 'model_var', 'n_e', 'n_s', 'model_param_const', 'dist'))
# Define UI for application that draws a histogram
ui <- fluidPage(

    # Application title
    titlePanel("View data from simulations"),

    # Sidebar with a slider input for number of bins 
    sidebarLayout(
        sidebarPanel(
          radioButtons(
            'model_type',
            'Which of variables changes',
            choices = c('u', 'i'),
            selected = 'u'),
          selectInput('model', 'Which model to show?', unique(df_folders$name)),
            # sliderInput("n_e",
            #             "number of electrons:",
            #             min = 2,
            #             max = 10,
            #             value = 6, step = 2),
            # sliderInput("u",
            #             "U parameter:",
            #             min = 2,
            #             max = 10,
            #             value = 4, step = 2),
            plotOutput('param_plot', click = 'plot1_click', width = '80%', height = '300px'),
            radioButtons('plotname', 'Which plot to plot', 
                         c('energy'='Energy_errors.svg', 
                           'density'='Densities.svg', 
                           'energy per site'='Energy_errors_per_site.svg',
                           'Hxc potential'='v_hxc_trend.svg'), )
        ),

        # Show a plot of the generated distribution
        mainPanel(
          htmlOutput("showInput"),

          fluidRow(imageOutput('imgplot'))
        )
    )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
    vals <- reactiveValues()
    
    observe(
      vals$path_plot <- paste(vals$folder, '/', input$plotname, sep=''))

    observeEvent(input$plot1_click, {vals$plot1_click <- input$plot1_click })
    observeEvent(input$model_type, {vals$plot1_click <- df_folders[FALSE, ]})
    observeEvent(vals$plot1_click, {
      df_selected <- nearPoints(vals$df2, input$plot1_click)
      vals$df_selected <- df_selected
      vals$folder <- df_selected[1, 'folder']
    })
    
    output$param_plot <- renderPlot({
      if (input$model_type == 'u'){const_var <- 'i'}
      else {const_var <- 'U'}
      df2 <- df_folders[(df_folders$name == input$model)&(df_folders$model_var == input$model_type), ]
      vals$df2 <- df2
      (ggplot(df2, aes(x=n_e, y=model_param_const))+
        geom_point(size=7, shape=1)+
          geom_point(data=vals$df_selected, color='red', shape=4, size=7)+
          theme_bw()+
          scale_y_continuous(breaks  = unique(df2$model_param_const), limits=c(min(df2$model_param_const - 0.5), max(df2$model_param_const + 0.5)))+
          scale_x_continuous(breaks  = unique(df2$n_e), limits=c(min(df2$n_e - 2), max(df2$n_e + 2)))+
          xlab('Number of electrons')+
          ylab(const_var)
      )
    })
    
    output$showInput <-  renderUI({
      if (input$model_type == 'u'){const_var <- 'i'}
      else {const_var <- 'U'}
      HTML(paste('<b>System data: </b><br/>System: ', vals$df_selected[1, 'name'],
                 "<br/>Number of electrons: ", vals$df_selected[1, 'n_e'],
                 "<br/>", const_var, ": ", vals$df_selected[1, 'model_param_const'],
                 "<br/> Path to file: ", vals$filename))
      
    })
    
    
    output$imgplot <- renderImage({
      # When input$n is 1, filename is www/images/image1.jpeg
      filename <- paste('C:/Users/tinc9/Documents/CNRS-offline/internship_project/results/', vals$path_plot
                                          , sep='')
      
      if ((grepl('NA', filename, fixed = TRUE))|(is.null(vals$folder))){
        filename <- 'C:/Users/tinc9/Documents/CNRS-offline/internship_project/results/default.svg'
      }
      vals$filename <-  filename
      
      # Return a list containing the filename
      list(src = filename, width = 1000)
    }, deleteFile = FALSE)
}

# Run the application 
shinyApp(ui = ui, server = server)
